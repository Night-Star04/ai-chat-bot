import os
import json
from typing import List, Optional
from colorama import init, Fore, Style

# 初始化 colorama (terminal color support)
init(autoreset=True)

help_text = f"""
{Fore.CYAN}=== 進階 OpenAI 聊天機器人 幫助 ==={Style.RESET_ALL}

{Fore.YELLOW}基本指令：{Style.RESET_ALL}
{Fore.GREEN}/help{Style.RESET_ALL}     - 顯示此幫助訊息
{Fore.GREEN}/clear{Style.RESET_ALL}    - 清除對話歷史
{Fore.GREEN}/quit{Style.RESET_ALL}     - 退出聊天機器人
{Fore.GREEN}/exit{Style.RESET_ALL}     - 退出聊天機器人

{Fore.YELLOW}進階指令：{Style.RESET_ALL}
{Fore.GREEN}/model <模型名稱>{Style.RESET_ALL}    - 更換 AI 模型
{Fore.GREEN}/models{Style.RESET_ALL}             - 顯示可用模型
{Fore.GREEN}/temp <數值>{Style.RESET_ALL}        - 設定溫度 (0-2)
{Fore.GREEN}/tokens <數值>{Style.RESET_ALL}      - 設定最大令牌數 (1-4096)
{Fore.GREEN}/prompt <數值>{Style.RESET_ALL}      - 設定機器人提示詞 (< 1000 字符)
{Fore.GREEN}/stats{Style.RESET_ALL}              - 顯示統計資訊
{Fore.GREEN}/settings{Style.RESET_ALL}           - 顯示當前設定
        """


def get_environment_variable(var_name: str,
                             default: Optional[str] = None,
                             show_warning: bool = False) -> Optional[str]:
    """
    獲取環境變數值
    
    從系統環境變數中獲取指定名稱的變數值，如果變數不存在，
    可以返回預設值或顯示警告訊息。
    
    Args:
        var_name (str): 環境變數名稱
        default (Optional[str], optional): 當環境變數不存在時的預設值。預設為 None。
        show_warning (bool, optional): 當環境變數不存在且沒有預設值時，是否顯示警告訊息。預設為 False。
    
    Returns:
        Optional[str]: 環境變數的值，如果不存在則返回預設值或 None
        
    Example:
        >>> get_environment_variable("HOME")
        "/home/user"
        >>> get_environment_variable("NONEXISTENT", "default_value")
        "default_value"
        >>> get_environment_variable("NONEXISTENT", show_warning=True)
        警告：環境變數 NONEXISTENT 未設定，請檢查您的環境設定
    """
    value = os.getenv(var_name)
    if value is None:
        if default is not None:
            return default
        else:
            if show_warning:
                print(
                    f"{Fore.YELLOW}警告：環境變數 {var_name} 未設定，請檢查您的環境設定{Style.RESET_ALL}"
                )
    return value


def load_models_from_json() -> List[str]:
    """從 models.json 檔案載入可用的模型列表"""
    try:
        models_file = "models.json"
        if os.path.exists(models_file):
            with open(models_file, 'r', encoding='utf-8') as f:
                models = json.load(f)
                if isinstance(models, list):
                    print(
                        f"{Fore.GREEN}已從 {models_file} 載入 {len(models)} 個模型{Style.RESET_ALL}"
                    )
                    return models
                else:
                    print(
                        f"{Fore.YELLOW}警告：{models_file} 格式不正確，使用預設模型列表{Style.RESET_ALL}"
                    )
        else:
            print(
                f"{Fore.YELLOW}警告：找不到 {models_file}，使用預設模型列表{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}載入模型列表失敗：{str(e)}，使用預設模型列表{Style.RESET_ALL}")

    # 如果載入失敗，返回預設模型列表
    return [
        "gpt-4",
        "gpt-4.1",
        "gpt-4o",
    ]


def print_help():
    """顯示幫助訊息"""
    print(help_text)
