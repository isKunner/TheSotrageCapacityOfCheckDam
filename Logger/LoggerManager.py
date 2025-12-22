import logging
import os


class LoggerManager:

    logger_name = "StroageCapacityCalculation"
    log_dir  = "logs"

    def __init__(self, logger_name=None, log_dir=None, log_info=None):

        # 初始化日志器（使用指定名称）
        if logger_name is not None:
            self.logger_name = logger_name
            LoggerManager.logger_name = logger_name
        if log_dir is not None:
            self.log_dir = log_dir
            LoggerManager.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.logger = self._init_logger()
        if log_info:
            self.logger.info(log_info)

    def _init_logger(self):

        logger = logging.getLogger(self.logger_name)
        logger.setLevel(logging.INFO)

        # 避免重复添加处理器
        if not logger.handlers:
            # 日志文件名按时间戳生成
            self.log_filename = f"{self.logger_name}.log"
            self.log_path = os.path.join(self.log_dir, self.log_filename)

            # 文件处理器
            file_handler = logging.FileHandler(self.log_path, encoding='utf-8')
            # 控制台处理器
            console_handler = logging.StreamHandler()

            # 日志格式
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )

            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            logger.addHandler(file_handler)
            logger.addHandler(console_handler)

        return logger

    @classmethod
    def get_logger(cls):
        logger = logging.getLogger(cls.logger_name)

        if not logger.handlers:
            cls_instance = cls(cls.logger_name, log_dir=cls.log_dir)
            return cls_instance.logger

        return logger

    def get_log_content(self):
        try:
            with open(self.log_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            error_msg = f"获取日志失败: {str(e)}"
            self.logger.error(error_msg)
            return error_msg

# 获取基本信息
def get_env_info(version=1.0):
    """Get environment information.

    Currently, only log the software version.
    """

    msg = r"""
           ___                           ___                ___                          
          (   )                         (   )              (   )                         
  .--.     | | .-.     .--.     .--.     | |   ___       .-.| |    .---.   ___ .-. .-.   
 /    \    | |/   \   /    \   /    \    | |  (   )     /   \ |   / .-, \ (   )   '   \  
|  .-. ;   |  .-. .  |  .-. ; |  .-. ;   | |  ' /      |  .-. |  (__) ; |  |  .-.  .-. ; 
|  |(___)  | |  | |  |  | | | |  |(___)  | |,' /       | |  | |    .'`  |  | |  | |  | | 
|  |       | |  | |  |  |/  | |  |       | .  '.       | |  | |   / .'| |  | |  | |  | | 
|  | ___   | |  | |  |  ' _.' |  | ___   | | `. \      | |  | |  | /  | |  | |  | |  | | 
|  '(   )  | |  | |  |  .'.-. |  '(   )  | |   \ \     | '  | |  ; |  ; |  | |  | |  | | 
'  `-' |   | |  | |  '  `-' / '  `-' |   | |    \ .    ' `-'  /  ' `-'  |  | |  | |  | | 
 `.__,'   (___)(___)  `.__.'   `.__,'   (___ ) (___)    `.__,'   `.__.'_. (___)(___)(___)
                                                                                         
                                                                                         
                                ___     ___                        ___            ___    
                               (   )   (   )                      (   )          (   )   
  .--.     .--.     .--.     .-.| |     | |   ___  ___    .--.     | |   ___      | |    
 /    \   /    \   /    \   /   \ |     | |  (   )(   )  /    \    | |  (   )     | |    
;  ,-. ' |  .-. ; |  .-. ; |  .-. |     | |   | |  | |  |  .-. ;   | |  ' /       | |    
| |  | | | |  | | | |  | | | |  | |     | |   | |  | |  |  |(___)  | |,' /        | |    
| |  | | | |  | | | |  | | | |  | |     | |   | |  | |  |  |       | .  '.        | |    
| |  | | | |  | | | |  | | | |  | |     | |   | |  | |  |  | ___   | | `. \       | |    
| '  | | | '  | | | '  | | | '  | |     | |   | |  ; '  |  '(   )  | |   \ \      |_|    
'  `-' | '  `-' / '  `-' / ' `-'  /     | |   ' `-'  /  '  `-' |   | |    \ .     .-.    
 `.__. |  `.__.'   `.__.'   `.__,'     (___)   '.__.'    `.__,'   (___ ) (___)   (   )   
 ( `-' ;                                                                          '-'    
  `.__.                                                                                  
    
    """
    msg += (f'\nVersion Information: {version}'
            f'\n\tSoftware Version: ArcGIS Pro 3.4.3')
    return msg