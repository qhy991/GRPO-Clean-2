from dataclasses import dataclass, field
import os
from typing import Optional
from datetime import datetime

@dataclass
class EnvConfig:
    """
    用于设置环境变量的配置。
    这些变量会在训练脚本 (train.py) 的早期被设置。
    """
    # Hugging Face 和网络代理配置
    hf_endpoint: Optional[str] = field(default=None, metadata={"help": "Hugging Face端点镜像，例如：'https://hf-mirror.com'。"})
    http_proxy: Optional[str] = field(default=None, metadata={"help": "HTTP代理服务器地址，例如：'http://user:pass@host:port'。"})
    https_proxy: Optional[str] = field(default=None, metadata={"help": "HTTPS代理服务器地址，例如：'http://user:pass@host:port'。"})

    # Weights & Biases 相关环境变量
    wandb_project: Optional[str] = field(default="VerilogGRPO-Enhanced", metadata={"help": "Weights & Biases 项目名称。"})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "Weights & Biases 实体（用户名或团队名）。如果为None，则使用 'wandb login' 时设置的默认实体。"})
    wandb_run_name_prefix: Optional[str] = field(default="enhanced-grpo-run", metadata={"help": "W&B运行名称的前缀。时间戳和关键参数会自动追加。"})
    wandb_disable: bool = field(default=False, metadata={"help": "设置为True以显式禁用W&B日志记录 (会设置 WANDB_DISABLED=true)。"})

    _generated_wandb_run_name: Optional[str] = field(init=False, repr=False, default=None)

    def __post_init__(self):
        print("INFO: Initializing EnvConfig and setting environment variables...")
        if self.hf_endpoint:
            os.environ['HF_ENDPOINT'] = self.hf_endpoint
            print(f"INFO: Set HF_ENDPOINT to {self.hf_endpoint}")
        if self.http_proxy:
            os.environ['http_proxy'] = self.http_proxy
            print(f"INFO: Set http_proxy to {self.http_proxy}")
        if self.https_proxy:
            os.environ['https_proxy'] = self.https_proxy
            print(f"INFO: Set https_proxy to {self.https_proxy}")

        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

        if self.wandb_disable:
            os.environ['WANDB_DISABLED'] = "true"
            print("INFO: W&B logging explicitly disabled via EnvConfig (WANDB_DISABLED=true).")
        else:
            if self.wandb_project:
                os.environ['WANDB_PROJECT'] = self.wandb_project

            if self.wandb_entity:
                os.environ['WANDB_ENTITY'] = self.wandb_entity

            if self.wandb_run_name_prefix:
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                self._generated_wandb_run_name = f"{self.wandb_run_name_prefix}-{timestamp}"
                if not os.getenv('WANDB_RUN_NAME'):
                    os.environ['WANDB_RUN_NAME'] = self._generated_wandb_run_name

        print(f"INFO: Effective W&B Environment:")
        print(f"  WANDB_PROJECT: {os.getenv('WANDB_PROJECT')}")
        print(f"  WANDB_ENTITY: {os.getenv('WANDB_ENTITY')}")
        print(f"  WANDB_RUN_NAME: {os.getenv('WANDB_RUN_NAME')}")
        print(f"  WANDB_DISABLED: {os.getenv('WANDB_DISABLED')}")
        print(f"  WANDB_API_KEY is SET: {'YES' if os.getenv('WANDB_API_KEY') else 'NO (relies on login/config)'}")
