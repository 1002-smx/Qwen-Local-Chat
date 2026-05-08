import torch
from transformers import AutoProcessor, AutoModelForCausalLM, AutoTokenizer
import warnings
import time
import sys
import json
import os
from datetime import datetime
from typing import List, Dict, Optional
from unittest.mock import MagicMock

# 创建一个假的 transformers_stream_generator 模块来避免导入错误
sys.modules['transformers_stream_generator'] = MagicMock()
sys.modules['transformers_stream_generator'].init_stream_support = MagicMock()

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_PATH = "./models"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HISTORY_DIR = "conversation_history"

# 创建历史记录目录
os.makedirs(HISTORY_DIR, exist_ok=True)

# 加载 processor（用于处理多模态输入，虽然这里只用文本）
processor = AutoProcessor.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="right"
)

# 直接加载 tokenizer（用于文本编码和解码）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="right"
)

# 设置 pad_token（如果不存在）
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

if DEVICE == "cuda":
    dtype = torch.float16
    offload_dir = "offload"
else:
    dtype = torch.float32
    offload_dir = None

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=dtype,
    device_map="auto" if DEVICE == "cuda" else "cpu",
    offload_folder=offload_dir
).eval()


def time_now():
    """获取计算机时间"""
    times = datetime.now()
    return f"{times}  {times.strftime('%A')}"


def SYSTEM_PROMPT():
    """实时更改人设，每次改为最新时间"""
    return (f"你是一个存在于用户电脑的AI助手，你需要认真回答用户的问题和接受用户的要求。"
            f"如果用户询问与时间有关的问题，请使用程序获取的用户的计算机时间，不要使用数据库里的时间，不用管时区之类的，时间：{time_now()}")

class ConversationManager:
    """对话管理器，支持多主题和对话记录保存"""

    def __init__(self, history_dir: str = HISTORY_DIR):
        self.history_dir = history_dir
        self.current_topic: Optional[str] = None
        self.topics: Dict[str, List[Dict]] = {}
        self.load_all_topics()

    def load_all_topics(self):
        """加载所有已保存的主题"""
        if not os.path.exists(self.history_dir):
            return

        for filename in os.listdir(self.history_dir):
            if filename.endswith('.json'):
                topic_name = filename[:-5]
                filepath = os.path.join(self.history_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.topics[topic_name] = json.load(f)
                except Exception as e:
                    print(f"加载主题 {topic_name} 失败: {e}")

    def save_topic(self, topic_name: str):
        """保存指定主题的对话记录"""
        if topic_name not in self.topics:
            return

        filepath = os.path.join(self.history_dir, f"{topic_name}.json")
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.topics[topic_name], f, ensure_ascii=False, indent=2)

    def create_topic(self, topic_name: str) -> bool:
        """创建新主题"""
        if topic_name in self.topics:
            print(f"主题 '{topic_name}' 已存在")
            return False

        self.topics[topic_name] = []
        self.current_topic = topic_name
        print(f"创建新主题: {topic_name}")
        return True

    def switch_topic(self, topic_name: str) -> bool:
        """切换到指定主题"""
        if topic_name not in self.topics:
            print(f"主题 '{topic_name}' 不存在，请先创建")
            return False

        self.current_topic = topic_name
        print(f"切换到主题: {topic_name}")
        return True

    def delete_topic(self, topic_name: str) -> bool:
        """删除主题"""
        if topic_name not in self.topics:
            print(f"主题 '{topic_name}' 不存在")
            return False

        filepath = os.path.join(self.history_dir, f"{topic_name}.json")
        if os.path.exists(filepath):
            os.remove(filepath)

        del self.topics[topic_name]

        if self.current_topic == topic_name:
            self.current_topic = None

        print(f"已删除主题: {topic_name}")
        return True

    def list_topics(self) -> List[str]:
        """列出所有主题"""
        return list(self.topics.keys())

    def add_message(self, role: str, content: str):
        """添加消息到当前主题"""
        if self.current_topic is None:
            print("请先创建或切换到一个主题！")
            return False

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }

        self.topics[self.current_topic].append(message)
        self.save_topic(self.current_topic)
        return True

    def build_chat_prompt(self, max_messages: Optional[int] = None) -> str:
        """构建Qwen正确的聊天格式prompt"""
        if self.current_topic is None or self.current_topic not in self.topics:
            return ""

        messages = self.topics[self.current_topic]
        if max_messages:
            messages = messages[-max_messages:]

        prompt = f"<|im_start|>system\n{SYSTEM_PROMPT()}<|im_end|>\n"

        # 添加历史对话
        for msg in messages:
            if msg["role"] == "user":
                prompt += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
            elif msg["role"] == "assistant":
                prompt += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"

        # 添加assistant开始标记
        prompt += "<|im_start|>assistant\n"

        return prompt

    def display_conversation(self):
        """显示当前对话记录"""
        if self.current_topic is None or self.current_topic not in self.topics:
            print("当前没有激活的主题！")
            return

        print(f"\n===== 主题: {self.current_topic} =====")
        for i, msg in enumerate(self.topics[self.current_topic], 1):
            role_name = "用户" if msg["role"] == "user" else "助手"
            print(f"\n[{i}] {role_name} ({msg['timestamp'][:19]}):")
            print(f"  回答: {msg['content']}")
        print("=" * 50)


def stream_generate(models, input_ids, tokenizer_obj, delay: float = 0.05) -> str:
    """流式生成回答"""
    generated_texts = ""
    current_ids = input_ids
    max_tokens = 512

    # 获取pad_token_id
    pad_token_id = tokenizer_obj.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer_obj.eos_token_id

    with torch.no_grad():
        for step in range(max_tokens):
            outputs = models.generate(
                current_ids,
                max_new_tokens=1,
                temperature=0.8,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.15,
                pad_token_id=pad_token_id,
                eos_token_id=tokenizer_obj.eos_token_id
            )

            new_token_id = outputs[0][-1:]
            current_ids = outputs
            new_token = tokenizer_obj.decode(new_token_id, skip_special_tokens=True)

            if not new_token or new_token in ['<|im_end|>', '<|endoftext|>', '']:
                break

            if new_token == '\n' and len(generated_texts) > 50:
                break

            print(new_token, end='', flush=True)
            generated_texts += new_token
            time.sleep(delay)

    print()
    return generated_texts


def show_help():
    """显示帮助信息"""
    print("=" * 60)
    print("命令列表:")
    print("  /help           - 显示此帮助信息")
    print("  /topics         - 列出所有主题")
    print("  /create 名称    - 创建新主题")
    print("  /switch 名称    - 切换到指定主题")
    print("  /delete 名称    - 删除主题")
    print("  /history        - 显示当前对话记录")
    print("  /clear          - 清空屏幕")
    print("  /times          - 获取当前时间")
    print("  /quit           - 退出程序")
    print(" 直接输入文本     - 与AI对话")
    print("=" * 60)


def main():
    """主函数"""
    conv_manager = ConversationManager()
    show_help()

    print("-" * 50)
    topics = conv_manager.list_topics()
    if topics:
        print("目前已保存的主题:")
        for i, topic in enumerate(topics, 1):
            marker = " [当前]" if topic == conv_manager.current_topic else ""
            print(f"  {i}. {topic}{marker}")
    else:
        print("暂无任何主题！")
    print("-" * 50)

    print(f"当前主题: {conv_manager.current_topic or '无'}")

    while True:
        try:
            user_input = input("输入: ").strip()

            if not user_input:
                continue

            # 处理命令
            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                cmd = parts[0].lower()
                arg = parts[1] if len(parts) > 1 else ""

                if cmd == '/quit':
                    print("正在保存所有对话...")
                    for topic in conv_manager.list_topics():
                        conv_manager.save_topic(topic)
                    print("退出成功！")
                    return  # 直接返回，不执行后续代码

                elif cmd == '/help':
                    show_help()

                elif cmd == '/topics':
                    topics = conv_manager.list_topics()
                    if topics:
                        print("已保存的主题:")
                        for i, topic in enumerate(topics, 1):
                            marker = " [当前]" if topic == conv_manager.current_topic else ""
                            print(f"  {i}. {topic}{marker}")
                    else:
                        print("暂无任何主题！")

                elif cmd == '/create':
                    if not arg:
                        print("请指定主题名称，例如: /create 工作讨论！")
                    else:
                        conv_manager.create_topic(arg)

                elif cmd == '/switch':
                    if not arg:
                        print("请指定主题名称，例如: /switch 工作讨论！")
                    else:
                        conv_manager.switch_topic(arg)

                elif cmd == '/delete':
                    if not arg:
                        print("请指定要删除的主题名称！")
                    else:
                        if arg == conv_manager.current_topic:
                            print("不能删除当前正在使用的主题，请先切换到其他主题！")
                        else:
                            conv_manager.delete_topic(arg)

                elif cmd == '/history':
                    conv_manager.display_conversation()

                elif cmd == '/clear':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("屏幕已清空！")

                elif cmd == '/times':
                    print(f"现在时间是：{time_now()}")

                else:
                    print(f"未知命令: {cmd}，输入 /help 查看帮助！")

                continue

            # 正常对话
            if user_input.lower() == 'quit':
                print("请输入 /quit 退出程序！")
                continue

            # 确保有一个激活的主题
            if conv_manager.current_topic is None:
                print("未选择主题，正在创建默认主题...")
                default_topic = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                conv_manager.create_topic(default_topic)

            # 保存用户输入
            conv_manager.add_message("user", user_input)

            # 构建聊天prompt
            prompt = conv_manager.build_chat_prompt(max_messages=20)

            # 编码输入
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(DEVICE)

            if DEVICE == "cpu":
                inputs = inputs.to(torch.float32)

            print("Qwen: ", end='', flush=True)

            # 生成回答
            generated_text = stream_generate(
                model,
                inputs.input_ids,
                tokenizer,
                delay=0.05
            )

            # 保存助手回答
            if generated_text:
                conv_manager.add_message("assistant", generated_text)
            else:
                conv_manager.add_message("assistant", "抱歉，我无法生成有效的回答。")

            print("-" * 50)

        except KeyboardInterrupt:
            print("正在保存并退出...")
            for topic in conv_manager.list_topics():
                conv_manager.save_topic(topic)
            print("已保存所有对话，退出成功！")
            return
        except Exception as e:
            print(f"发生错误: {e}")
            import traceback
            traceback.print_exc()
            print("请重试或输入 /help 查看帮助！")


if __name__ == "__main__":
    main()
