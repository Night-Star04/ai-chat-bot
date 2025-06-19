from typing import Dict, Literal, Any
from colorama import init, Fore, Style

init(autoreset=True)


class ModeConfiguration:
    """
    模型配置類，用於管理聊天機器人的設定參數
    
    此類提供了一種結構化的方式來存儲和管理 OpenAI API 的各種配置參數，
    包括模型選擇、令牌限制、溫度設定和系統提示。
    
    Attributes:
        model (str): 使用的 OpenAI 模型名稱
        max_tokens (int): API 回應的最大令牌數
        temperature (float): 回應的隨機性程度 (0.0-2.0)
        system_prompt (str | None): 系統提示訊息
    """
    model: str
    max_tokens: int
    temperature: float
    system_prompt: str | None

    ConfigurationKey = Literal["model", "max_tokens", "temperature",
                               "system_prompt"]

    def __init__(self, model: str, max_tokens: int, temperature: float,
                 system_prompt: str | None):
        """
        初始化模型配置
        
        Args:
            model (str): OpenAI 模型名稱
            max_tokens (int): 最大令牌數，必須為正整數
            temperature (float): 溫度值，範圍 0.0-2.0
            system_prompt (str | None): 系統提示，可為空
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.system_prompt = system_prompt

    def __getitem__(self, key: ConfigurationKey) -> Any:
        """
        允許通過鍵訪問配置值
        
        實現字典樣式的屬性訪問，使配置對象可以像字典一樣使用。
        
        Args:
            key (ConfigurationKey): 配置鍵名，必須是有效的配置項
            
        Returns:
            Any: 對應配置項的值
            
        Raises:
            KeyError: 當提供的鍵不是有效配置項時
            
        Example:
            >>> config = ModeConfiguration("gpt-4", 1024, 0.7, "Hello")
            >>> model = config["model"]  # 返回 "gpt-4"
        """
        if key == "model":
            return self.model
        elif key == "max_tokens":
            return self.max_tokens
        elif key == "temperature":
            return self.temperature
        elif key == "system_prompt":
            return self.system_prompt
        else:
            raise KeyError(f"未知的設定鍵：{key}")

    def __setitem__(self, key: ConfigurationKey, value: Any):
        """
        允許通過鍵設置配置值
        
        實現字典樣式的屬性設置，使配置對象可以像字典一樣修改。
        
        Args:
            key (ConfigurationKey): 配置鍵名，必須是有效的配置項
            value (Any): 要設置的新值
            
        Raises:
            KeyError: 當提供的鍵不是有效配置項時
            
        Example:
            >>> config = ModeConfiguration("gpt-4", 1024, 0.7, "Hello")
            >>> config["temperature"] = 0.5  # 修改溫度值
        """
        if key == "model":
            self.model = value
        elif key == "max_tokens":
            self.max_tokens = value
        elif key == "temperature":
            self.temperature = value
        elif key == "system_prompt":
            self.system_prompt = value
        else:
            raise KeyError(f"未知的設定鍵：{key}")

    def print(self):
        """
        打印當前配置資訊
        
        以格式化的方式顯示所有配置參數的當前值，包括模型名稱、
        最大令牌數、溫度設定和系統提示。使用彩色輸出增強可讀性。
        
        Note:
            系統提示為空時會顯示 "未設定"
        """
        print(f"""
{Fore.CYAN}=== 當前設定 ==={Style.RESET_ALL}
模型：{self.model}
最大令牌數：{self.max_tokens}
溫度：{self.temperature}
系統提示：{self.system_prompt or "未設定"}
        """)

    def to_dict(self) -> Dict[ConfigurationKey, Any]:
        """
        將配置轉換為字典格式
        
        返回一個包含所有配置參數的字典，便於序列化或存儲。
        
        Returns:
            Dict[str, Any]: 包含所有配置參數的字典
        """
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "system_prompt": self.system_prompt
        }

    @staticmethod
    def from_dict(data: Dict[ConfigurationKey, Any]) -> "ModeConfiguration":
        """
        從字典創建配置對象
        
        靜態方法，用於從字典格式的配置數據創建 ModeConfiguration 實例。
        
        Args:
            data (Dict[str, Any]): 包含配置參數的字典
            
        Returns:
            ModeConfiguration: 新的配置對象
            
        Raises:
            KeyError: 當字典中缺少必要的配置項時
        """
        return ModeConfiguration(model=data["model"],
                                 max_tokens=data["max_tokens"],
                                 temperature=data["temperature"],
                                 system_prompt=data.get("system_prompt", None))
