import json
import sys
from datetime import datetime
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
from colorama import init, Fore, Style
from ModeConfiguration import ModeConfiguration
from utils import get_environment_variable, load_models_from_json, print_help

# 初始化 colorama (terminal color support)
init(autoreset=True)


class ChatBot:
    """
    這是一個功能完整的聊天機器人實現，支援多種模型切換、對話歷史管理、
    統計資訊追蹤、配置調整等進階功能。提供命令行界面供用戶互動。
    
    主要功能:
    - 支援多種 OpenAI 模型
    - 對話歷史保存和載入
    - 即時配置調整
    - 統計資訊追蹤
    - 豊富的命令系統
    - 彩色終端輸出
    
    Attributes:
        api_key (str): OpenAI API 金鑰
        client (OpenAI): OpenAI 客戶端實例
        available_models (List[str]): 可用的模型列表
        settings (ModeConfiguration): 當前配置設定
        conversation_history (List[Dict[str, str]]): 對話歷史記錄
        stats (Dict): 統計資訊
    """

    def __init__(self):
        """
        初始化進階聊天機器人
        
        執行以下初始化步驟：
        1. 載入環境變數
        2. 驗證 OpenAI API 金鑰
        3. 初始化 OpenAI 客戶端
        4. 載入可用模型列表
        5. 設定預設配置
        6. 初始化對話歷史和統計資訊
        
        Raises:
            SystemExit: 當 API 金鑰未設定或模型配置無效時
        """
        load_dotenv()

        # 從環境變數讀取 API 金鑰並驗證
        self.api_key = get_environment_variable("OPENAI_API_KEY")
        if not self.api_key:
            print(f"{Fore.RED}錯誤：請設定 OPENAI_API_KEY 環境變數{Style.RESET_ALL}")
            print("請在 .env 檔案中設定您的 OpenAI API 金鑰")
            sys.exit(1)
        if self.api_key.startswith("sk-") is False or len(self.api_key) < 20:
            print(f"{Fore.RED}錯誤：無效的 OpenAI API 金鑰，請檢查您的設定{Style.RESET_ALL}")
            sys.exit(1)

        # 初始化 OpenAI 客戶端
        self.client = OpenAI(api_key=self.api_key)

        # 從 models.json 載入可用的模型
        self.available_models = load_models_from_json()

        # 預設設定
        self.settings = self.load_model_configuration()

        # 對話歷史
        self.conversation_history: List[Dict[str, str]] = [{
            "role":
            "developer",
            "content":
            self.settings["system_prompt"] or ""
        }]

        # 對話統計
        self.stats = {
            "total_messages": 0,
            "total_tokens_used": 0,
            "session_start": datetime.now()
        }

    def load_model_configuration(self) -> ModeConfiguration:
        """
        載入並驗證模型配置
        
        從環境變數中讀取各種配置參數，並進行驗證。如果參數無效，
        程式將顯示錯誤訊息並退出。
        
        環境變數參數:
        - OPENAI_MODEL: 要使用的模型名稱 (預設: "gpt-4o")
        - OPENAI_MAX_TOKENS: 最大令牌數 (預設: "1024")
        - OPENAI_TEMPERATURE: 溫度值 (預設: "0.7")
        - OPENAI_SYSTEM_PROMPT: 系統提示 (預設: 中文助手提示)
        
        Returns:
            ModeConfiguration: 包含所有驗證過的配置參數的對象
            
        Raises:
            SystemExit: 當任何配置參數無效時退出程式
            
        Validation Rules:
        - 模型必須在可用模型列表中
        - max_tokens 必須是正整數
        - temperature 必須在 0-2 範圍內
        - system_prompt 長度不應超過 1000 字元
        """
        model = get_environment_variable("OPENAI_MODEL",
                                         default="gpt-4o",
                                         show_warning=False)
        if model not in self.available_models:
            print(
                f"{Fore.RED}錯誤：指定的模型 {model} 不在可用模型列表中，請檢查 models.json 或環境變數設定{Style.RESET_ALL}"
            )
            print(f"可用模型：{', '.join(self.available_models)}")
            sys.exit(1)

        max_tokens = get_environment_variable("OPENAI_MAX_TOKENS",
                                              default="1024",
                                              show_warning=False)
        if max_tokens and (not max_tokens.isdigit() or int(max_tokens) <= 0):
            print(
                f"{Fore.RED}錯誤：OPENAI_MAX_TOKENS 必須是一個正整數，請檢查您的環境變數設定{Style.RESET_ALL}"
            )
            sys.exit(1)
        temperature = get_environment_variable("OPENAI_TEMPERATURE",
                                               default="0.7",
                                               show_warning=False)
        if temperature and (not temperature.replace('.', '', 1).isdigit()
                            or not (0 <= float(temperature) <= 2)):
            print(
                f"{Fore.RED}錯誤：OPENAI_TEMPERATURE 必須在 0 到 2 之間，請檢查您的環境變數設定{Style.RESET_ALL}"
            )
            sys.exit(1)

        system_prompt = get_environment_variable(
            "OPENAI_SYSTEM_PROMPT",
            default="你是一個智慧型助手，請根據使用者的問題提供清楚、有幫助的回答。若問題不夠明確，請詢問更多資訊。",
            show_warning=False)
        if not system_prompt:
            print(
                f"{Fore.YELLOW}警告：OPENAI_SYSTEM_PROMPT 未設定，使用預設系統提示{Style.RESET_ALL}"
            )
        elif len(system_prompt) > 1000:
            print(
                f"{Fore.YELLOW}警告：OPENAI_SYSTEM_PROMPT 過長，建議不超過 1000 字元，將使用前 1000 字元{Style.RESET_ALL}"
            )
            system_prompt = system_prompt[:1000]
            print(f"{Fore.GREEN}系統提示已設定為：{system_prompt}{Style.RESET_ALL}")

        return ModeConfiguration(
            model=model,
            max_tokens=int(max_tokens) if max_tokens else 1024,
            temperature=float(temperature) if temperature else 0.7,
            system_prompt=system_prompt)

    def save_conversation(self, filename: Optional[str] = None):
        """
        儲存對話歷史到 JSON 檔案
        
        將當前的對話歷史、配置設定和統計資訊保存到 JSON 檔案中。
        如果未指定檔案名稱，將使用時間戳自動生成檔案名稱。
        
        Args:
            filename (Optional[str]): 儲存的檔案名稱，如果為 None 則自動生成
                格式為 "conversation_YYYYMMDD_HHMMSS.json"
                
        JSON Structure:
            保存的檔案包含以下結構：
            - timestamp: 保存時間戳（ISO 格式）
            - settings: 當前配置設定（使用 to_dict() 方法序列化）
            - conversation: 對話歷史（排除系統提示）
            - stats: 統計資訊（datetime 對象轉換為 ISO 字串）
                
        Note:
            - 系統提示不會被保存到對話歷史中，但會保存在 settings 中
            - datetime 對象會自動轉換為 ISO 格式字串以支援 JSON 序列化
            - 使用 UTF-8 編碼確保中文字符正確保存
            
        Error Handling:
            發生錯誤時會顯示錯誤訊息但不會中斷程式執行
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

        conversation_data = {
            "timestamp": datetime.now().isoformat(),
            "settings": self.settings.to_dict(),
            "conversation": self.conversation_history[1:],  # 排除系統提示
            "stats": {
                "total_messages": self.stats["total_messages"],
                "total_tokens_used": self.stats["total_tokens_used"],
                "session_start": self.stats["session_start"].isoformat()
            }
        }

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            print(f"{Fore.GREEN}對話已儲存到：{filename}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}儲存對話失敗：{str(e)}{Style.RESET_ALL}")

    def load_conversation(self, filename: str):
        """
        從 JSON 檔案載入對話歷史
        
        讀取之前保存的對話檔案，恢復對話歷史到當前會話中。
        支援載入對話歷史和配置設定，並提供用戶確認機制。
        
        Args:
            filename (str): 要載入的對話檔案路徑
            
        Loading Process:
        1. 解析 JSON 檔案並使用 from_dict() 方法恢復配置
        2. 詢問用戶是否要覆蓋當前對話歷史
        3. 詢問用戶是否要更新系統配置設定
        4. 重建對話歷史並顯示載入結果
            
        User Interaction:
        - 對話歷史覆蓋確認：防止意外丟失當前對話
        - 系統設定更新確認：允許用戶選擇是否使用載入的配置
        - 詳細的操作反饋和統計資訊
            
        JSON Structure Expected:
            檔案應包含以下結構（由 save_conversation 生成）：
            - settings: 配置設定字典
            - conversation: 對話歷史陣列
            - stats: 統計資訊（可選）
            - timestamp: 保存時間戳（可選）
            
        Note:
            - 載入的系統提示會用於重建對話歷史的第一條訊息
            - 用戶可以選擇只載入對話歷史而保留當前系統設定
            - 載入操作提供完整的確認和回滾機制
            
        Error Handling:
            如果檔案不存在、格式錯誤或缺少必要欄位，會顯示錯誤訊息但不會中斷程式
        """
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)

            setting = ModeConfiguration.from_dict(data["settings"])

            # 重建對話歷史
            self.conversation_history = [{
                "role": "system",
                "content": setting.system_prompt or ""
            }]

            if len(self.conversation_history) > 1:
                print(
                    f"{Fore.YELLOW}警告：載入的對話歷史將覆蓋當前對話歷史，是否繼續?{Style.RESET_ALL}")
                confirm = input(
                    f"{Fore.YELLOW}請輸入 'yes' 確認繼續，或按 Enter 取消：{Style.RESET_ALL}"
                ).strip().lower()
                if confirm != 'yes':
                    print(f"{Fore.YELLOW}載入已取消{Style.RESET_ALL}")
                    return

            self.conversation_history.extend(data["conversation"])

            print(
                f"{Fore.YELLOW}警告：載入的系統設定將覆蓋當前設定，請確認是否需要更新系統設定？{Style.RESET_ALL}"
            )
            confirm = input(
                f"{Fore.YELLOW}請輸入 'yes' 確認更新系統設定，或按 Enter 保留當前設定：{Style.RESET_ALL}"
            ).strip().lower()
            if confirm == 'yes':
                self.settings = setting
                print(
                    f"{Fore.GREEN}系統設定已更新為：{self.settings.to_dict()}{Style.RESET_ALL}"
                )
            else:
                print(f"{Fore.YELLOW}系統設定保留為當前設定{Style.RESET_ALL}")

            print(f"{Fore.GREEN}對話已從 {filename} 載入{Style.RESET_ALL}")
            print(f"載入了 {len(data['conversation'])} 條訊息")

        except Exception as e:
            print(f"{Fore.RED}載入對話失敗：{str(e)}{Style.RESET_ALL}")

    def adjust_settings(self, setting: ModeConfiguration.ConfigurationKey,
                        value):
        """
        動態調整聊天機器人的配置設定
        
        允許在執行時修改各種配置參數，包括模型選擇、令牌限制、
        溫度設定和系統提示。每個設定都有相應的驗證邏輯。
        
        Args:
            setting (ModeConfiguration.ConfigurationKey): 要調整的配置項名稱
                可選值: "model", "max_tokens", "temperature", "system_prompt"
            value: 新的配置值，類型取決於配置項
            
        Validation Rules:
            - model: 必須在可用模型列表中
            - max_tokens: 必須是正整數
            - temperature: 必須在 0.0-2.0 範圍內
            - system_prompt: 不能為空，長度限制 1000 字元
            
        Note:
            - 修改系統提示時會同時更新對話歷史中的系統訊息
            - 所有更改都會立即生效
            - 無效的值會顯示錯誤訊息但不會中斷程式
        """
        if setting == "model":
            if value in self.available_models:
                if value == self.settings["model"]:
                    print(
                        f"{Fore.YELLOW}已經在使用模型：{value}，無需更換{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}正在切換模型，請稍候...{Style.RESET_ALL}")
                self.settings["model"] = value
                print(f"{Fore.GREEN}已切換到模型：{value}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}不支援的模型：{value}{Style.RESET_ALL}")
                print(f"可用模型：{', '.join(self.available_models)}")

        elif setting == "max_tokens":
            try:
                tokens = int(value)
                if tokens > 0:
                    self.settings["max_tokens"] = tokens
                    print(f"{Fore.GREEN}最大令牌數已設定為：{tokens}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}令牌數必須大於 0{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}無效的令牌數值{Style.RESET_ALL}")

        elif setting == "temperature":
            try:
                temp = float(value)
                if 0 <= temp <= 2:
                    self.settings["temperature"] = temp
                    print(f"{Fore.GREEN}溫度已設定為：{temp}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}溫度必須在 0-2 之間{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}無效的溫度值{Style.RESET_ALL}")

        elif setting == "system_prompt":
            if value:
                if len(value) > 1000:
                    print(f"{Fore.YELLOW}系統提示過長，將使用前 1000 字元{Style.RESET_ALL}")
                    value = value[:1000]
                self.settings["system_prompt"] = value
                # 更新對話歷史中的系統提示
                self.conversation_history[0]["content"] = value
                print(f"{Fore.GREEN}系統提示已更新{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}系統提示不能為空{Style.RESET_ALL}")

        else:
            print(f"{Fore.RED}未知的設定：{setting}{Style.RESET_ALL}")

    def show_settings(self):
        """
        顯示當前所有配置設定
        
        調用 ModeConfiguration 的 print 方法來顯示當前的所有配置參數，
        包括模型名稱、最大令牌數、溫度設定和系統提示。
        """
        self.settings.print()

    def show_stats(self):
        """
        顯示詳細的統計資訊
        
        展示當前會話的統計數據，包括訊息數量、令牌使用量、
        會話持續時間等資訊。對於追蹤 API 使用情況很有用。
        
        Statistics Displayed:
        - 總訊息數：用戶和 AI 的訊息總數
        - 預估使用令牌：API 調用的總令牌消耗
        - 會話開始時間：當前會話的開始時間
        - 會話持續時間：從開始到現在的時間長度
        - 當前對話長度：對話歷史中的訊息數量（不含系統提示）
        """
        session_duration = datetime.now() - self.stats["session_start"]
        print(f"""
{Fore.CYAN}=== 對話統計 ==={Style.RESET_ALL}
總訊息數：{self.stats['total_messages']}
預估使用令牌：{self.stats['total_tokens_used']}
會話開始時間：{self.stats['session_start'].strftime('%Y-%m-%d %H:%M:%S')}
會話持續時間：{str(session_duration).split('.')[0]}
當前對話長度：{len(self.conversation_history) - 1} 條訊息
        """)

    def chat(self, message: str) -> str:
        """
        發送訊息給 AI 並獲取回應
        
        這是聊天機器人的核心功能，處理與 OpenAI API 的交互。
        會自動管理對話歷史並更新統計資訊。
        
        Args:
            message (str): 用戶輸入的訊息內容
            
        Returns:
            str: AI 的回應內容，如果發生錯誤則返回錯誤訊息
            
        Process Flow:
        1. 將用戶訊息添加到對話歷史
        2. 調用 OpenAI API 獲取回應
        3. 將 AI 回應添加到對話歷史
        4. 更新統計資訊（訊息數和令牌使用量）
        5. 返回 AI 回應
        
        Error Handling:
            任何 API 調用異常都會被捕獲並返回錯誤訊息字串
            
        Note:
            - 使用當前配置的模型、溫度和令牌限制
            - 自動維護完整的對話上下文
            - 統計資訊會即時更新
        """
        try:
            # 將用戶訊息添加到對話歷史
            self.conversation_history.append({
                "role": "user",
                "content": message
            })

            # 呼叫 OpenAI API
            response = self.client.responses.create(
                model=self.settings["model"],
                input=self.conversation_history,  # type: ignore
                temperature=self.settings["temperature"],
                max_output_tokens=self.settings.max_tokens)

            # 獲取助手的回應
            assistant_message = response.output_text

            # 將助手回應添加到對話歷史
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })

            # 更新統計
            self.stats["total_messages"] += 1
            if hasattr(response, 'usage') and response.usage:
                self.stats["total_tokens_used"] += response.usage.total_tokens

            return assistant_message

        except Exception as e:
            return f"錯誤：{str(e)}"

    def clear_history(self):
        """
        清除對話歷史但保留系統提示
        
        重置對話歷史到初始狀態，只保留系統提示訊息。
        這對於開始新話題或重置對話上下文很有用。
        
        Note:
            - 系統提示會被保留並使用當前配置
            - 統計資訊不會被重置
            - 操作不可撤銷
        """
        self.conversation_history = [{
            "role":
            "developer",
            "content":
            self.settings["system_prompt"] or ""
        }]
        print(f"{Fore.YELLOW}對話歷史已清除{Style.RESET_ALL}")

    def handle_command(self, user_input: str) -> bool:
        """
        處理用戶輸入的指令
        
        解析並執行以 '/' 開頭的命令。這是聊天機器人的命令系統核心，
        提供豐富的互動功能和配置管理。
        
        支援的命令：
        - /help: 顯示詳細的幫助資訊和使用指南
        - /clear: 清除對話歷史（保留系統提示）
        - /stats: 顯示詳細的統計資訊
        - /settings: 顯示當前所有配置設定
        - /models: 列出所有可用的模型（標示當前使用的模型）
        - /model <model_name>: 切換到指定的模型
        - /temp <value>: 設定溫度值（0.0-2.0）
        - /tokens <value>: 設定最大令牌數（正整數）
        - /prompt <text>: 設定系統提示內容
        - /save [filename]: 儲存對話歷史（可選指定檔名）
        - /load <filename>: 載入之前儲存的對話歷史
        - /quit, /exit: 優雅退出程式
        
        Args:
            user_input (str): 用戶輸入的完整字串
            
        Returns:
            bool: 如果輸入是有效命令則返回 True，否則返回 False
            
        Command Processing:
        1. 解析命令和參數（以空格分隔）
        2. 將命令轉換為小寫進行匹配
        3. 驗證必要參數的存在
        4. 執行對應的功能或顯示錯誤訊息
        
        Error Handling:
        - 未知命令：顯示錯誤訊息
        - 缺少必要參數：提示用戶提供所需參數
        - 參數驗證失敗：在各自的方法中處理
        
        Note:
            - 命令不區分大小寫
            - 參數用空格分隔，支援帶空格的參數
            - /quit 和 /exit 命令會直接終止程式
            - 所有配置更改都會立即生效
        """
        user_input_tokens = user_input.strip().split(maxsplit=1)

        command = user_input_tokens[0].lower()
        if len(user_input_tokens) > 1:
            args = user_input_tokens[1].strip()
        else:
            args = ""

        if not command.startswith('/'):
            return False

        if command == '/help':
            print_help()
        elif command == '/clear':
            self.clear_history()
        elif command == '/stats':
            self.show_stats()
        elif command == '/settings':
            self.show_settings()
        elif command == '/models':
            print(f"{Fore.CYAN}可用模型：{Style.RESET_ALL}")
            for model in self.available_models:
                if model == self.settings["model"]:
                    marker = " (當前)"
                else:
                    marker = ""

                print(f"  - {model}{marker}")
        elif command == '/model':
            if args:
                self.adjust_settings("model", args)
            else:
                print(f"{Fore.RED}請提供模型名稱{Style.RESET_ALL}")
        elif command == '/temp':
            if args:
                self.adjust_settings("temperature", args)
            else:
                print(f"{Fore.RED}請提供溫度值{Style.RESET_ALL}")
        elif command == '/tokens':
            if args:
                self.adjust_settings("max_tokens", args)
            else:
                print(f"{Fore.RED}請提供令牌數值{Style.RESET_ALL}")
        elif command == '/prompt':
            if args:
                self.adjust_settings("system_prompt", args)
            else:
                print(f"{Fore.RED}請提供系統提示內容{Style.RESET_ALL}")
        elif command == '/save':
            self.save_conversation(args if args else None)
        elif command == '/load':
            if args:
                self.load_conversation(args)
            else:
                print(f"{Fore.RED}請提供要載入的檔案名稱{Style.RESET_ALL}")
        elif command in ['/quit', '/exit']:
            print(f"{Fore.YELLOW}再見！感謝使用聊天機器人{Style.RESET_ALL}")
            sys.exit(0)
        else:
            print(f"{Fore.RED}未知指令：{command}{Style.RESET_ALL}")

        return True

    def run(self):
        """
        執行聊天機器人的主要互動迴圈
        
        這是聊天機器人的主要運行方法，實現了完整的用戶互動流程。
        包括命令處理、AI 對話、錯誤處理和用戶體驗優化。
        
        Main Loop Features:
        - 持續接收用戶輸入
        - 自動識別和處理命令
        - 與 AI 進行對話交互
        - 提供視覺反饋和狀態提示
        - 優雅處理中斷和異常
        
        User Experience:
        - 彩色提示符和輸出
        - "AI 思考中..." 的等待提示
        - 清晰的分隔線
        - Ctrl+C 優雅退出
        
        Error Handling:
        - KeyboardInterrupt: 用戶按 Ctrl+C 時優雅退出
        - 一般異常: 顯示錯誤訊息但繼續運行
        
        Note:
            這個方法會持續運行直到用戶輸入退出命令或按 Ctrl+C
        """
        print(f"{Fore.CYAN}=== OpenAI 聊天機器人 ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}輸入 /help 查看可用指令{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}輸入 /quit 或 /exit 退出程式{Style.RESET_ALL}")
        print("-" * 50)

        while True:
            try:
                # 獲取用戶輸入
                user_input = input(f"{Fore.BLUE}您：{Style.RESET_ALL}").strip()

                # 檢查空輸入
                if not user_input:
                    continue

                # 處理指令
                if self.handle_command(user_input):
                    # 如果是指令，則不進行 AI 回應
                    continue

                # 顯示思考中
                print(f"{Fore.YELLOW}AI 思考中...{Style.RESET_ALL}",
                      end="",
                      flush=True)

                # 獲取 AI 回應
                response = self.chat(user_input)

                # 清除思考中訊息並顯示回應
                print(f"\r{Fore.GREEN}AI：{Style.RESET_ALL}{response}")
                print("-" * 50)

            except KeyboardInterrupt:
                print(f"\n{Fore.YELLOW}程式被中斷，再見！{Style.RESET_ALL}")
                break
            except Exception as e:
                print(f"{Fore.RED}發生錯誤：{str(e)}{Style.RESET_ALL}")


def main():
    """
    程式主入口點
    
    負責創建 ChatBot 實例並啟動主運行迴圈。
    包含頂層的異常處理，確保程式啟動失敗時能提供清晰的錯誤資訊。
    
    Process:
    1. 創建 ChatBot 實例（會進行初始化驗證）
    2. 啟動主運行迴圈
    3. 處理任何啟動階段的異常
    
    Error Handling:
        如果聊天機器人無法啟動（如 API 金鑰錯誤、配置問題等），
        會顯示錯誤訊息並以錯誤碼 1 退出程式。
        
    Note:
        這個函數只在檔案被直接執行時才會被調用
    """
    try:
        # 建立並執行聊天機器人
        bot = ChatBot()
        bot.run()
    except Exception as e:
        print(f"{Fore.RED}無法啟動聊天機器人：{str(e)}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()
