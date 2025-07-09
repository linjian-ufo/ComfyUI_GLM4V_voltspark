import torch
from PIL import Image
import numpy as np
from transformers import AutoProcessor, Glm4vForConditionalGeneration, BitsAndBytesConfig
import folder_paths
import re
import os
import time
import threading
import glob
from pathlib import Path

class Glm4vNode:
    def __init__(self):
        # 初始化模型相关变量
        self.model = None  # 存储加载的模型
        self.processor = None  # 存储模型处理器
        self.device = None  # 存储当前使用的设备
        self.unload_timer = None  # 定时卸载模型的计时器
        self.current_model_name = None  # 当前加载的模型名称

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像
                "prompt": ("STRING", {  # 输入提示文本
                    "multiline": True,
                    "default": "describe this image,Describe in long sentence form, without using Markdown format."
                }),
                "model_name": (  # 添加模型选择下拉菜单
                    ["glmv4_4bit", "GLM-4.1V-9B-Thinking"],
                    {"default": "glmv4_4bit"}
                ),
                "unload_policy": (  # 模型卸载策略
                    ["Always", "Never", "After 1 min", "After 2 mins", "After 5 mins", "After 10 mins"],
                    {"default": "Always"}
                )
            },
        }

    RETURN_TYPES = ("STRING",)  # 返回类型为字符串
    RETURN_NAMES = ("description",)  # 返回值名称
    FUNCTION = "generate"  # 执行函数名
    CATEGORY = "GLM4V"  # 节点分类

    def tensor_to_pil(self, tensor):
        """将torch张量(B, H, W, C)转换为PIL图像"""
        # 如果张量有4个维度且第一个维度为1，则去掉第一个维度
        if tensor.ndim == 4 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        # 将浮点数[0, 1]转换为整数[0, 255]
        image_np = (tensor.cpu().numpy() * 255).astype(np.uint8)
        # 创建并返回PIL图像
        return Image.fromarray(image_np)

    def _determine_device(self):
        """确定使用的设备"""
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            return "cuda"  # 返回CUDA设备
        else:
            # 如果没有CUDA设备，抛出错误
            raise RuntimeError("GLM-4V (4bit) 仅支持 CUDA，未检测到可用的 CUDA 设备。")

    def _get_model_size(self, model_path, model_name):
        """获取模型大小估计"""
        # 根据不同模型返回不同的大小估计
        if model_name == "glmv4_4bit":
            # GLM-4V-9B 4bit版本大约6.5GB
            return 6.7 * 1024 * 1024 * 1024  # 6.7 GB
        elif model_name == "GLM-4.1V-9B-Thinking":
            # GLM-4.1V-9B-Thinking版本可能更大
            return 8.0 * 1024 * 1024 * 1024  # 8.0 GB
        else:
            # 默认大小
            return 6.7 * 1024 * 1024 * 1024  # 6.7 GB

    def _load_model(self, model_name):
        """加载指定的模型"""
        # 确定要使用的设备
        new_device = self._determine_device()
        
        # 检查是否需要重新加载模型
        if (self.model is not None and 
            self.processor is not None and 
            self.device == new_device and 
            self.current_model_name == model_name):
            print(f"GLM-4V: 模型 {model_name} 已加载，无需重新加载。")
            return  # 模型已在正确设备上加载且是正确的模型
        
        # 如果设备改变或模型改变，先卸载当前模型
        if (self.model is not None and 
            (self.device != new_device or self.current_model_name != model_name)):
            print(f"GLM-4V: 设备从 {self.device} 改变为 {new_device} 或模型从 {self.current_model_name} 改变为 {model_name}。重新加载模型。")
            self.unload_model()
        
        # 设置当前设备和模型名称
        self.device = new_device
        self.current_model_name = model_name
        print(f"GLM-4V: 使用设备: {self.device}, 加载模型: {model_name}")

        # 构建模型路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        comfyui_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
        model_path = os.path.join(comfyui_dir, "models", model_name)

        # 检查模型目录是否存在
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"模型目录未找到: {model_path}")

        print(f"正在从以下路径加载 GLM-4V 模型: {model_path}")
        
        try:
            # 加载处理器
            print("正在加载模型处理器...")
            self.processor = AutoProcessor.from_pretrained(
                model_path, 
                use_fast=True, 
                trust_remote_code=True
            )
            print("模型处理器加载成功。")

            # 设置模型加载参数
            from_pretrained_kwargs = {
                "pretrained_model_name_or_path": model_path,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.bfloat16,
                "trust_remote_code": True,
            }

            # 只支持 CUDA，进行显存管理
            try:
                # 清空CUDA缓存
                torch.cuda.empty_cache()
                print("已清空CUDA缓存。")
                
                # 获取可用显存信息
                free_vram, total_vram = torch.cuda.mem_get_info()
                model_size_bytes = self._get_model_size(model_path, model_name)
                print(f"GLM-4V: 可用显存: {free_vram / (1024**3):.2f}GB, "
                      f"总显存: {total_vram / (1024**3):.2f}GB, "
                      f"估计模型大小: {model_size_bytes / (1024**3):.2f}GB")

                # 根据显存情况决定加载策略
                if free_vram < model_size_bytes:
                    print("GLM-4V: 显存不足，使用量化配置和自动设备映射。")
                    # 配置4bit量化
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        llm_int8_enable_fp32_cpu_offload=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    from_pretrained_kwargs["quantization_config"] = quant_config
                    from_pretrained_kwargs["device_map"] = "auto"
                    print("已设置量化配置和自动设备映射。")
                else:
                    print("GLM-4V: 显存充足，模型全部加载到 GPU。")
                    from_pretrained_kwargs["device_map"] = "cuda:0"
                    
            except Exception as e:
                print(f"GLM-4V: 显存检测失败，回退到自动设备映射。错误: {e}")
                # 回退到安全的量化配置
                quant_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    llm_int8_enable_fp32_cpu_offload=True,
                    bnb_4bit_quant_type="nf4"
                )
                from_pretrained_kwargs["device_map"] = "auto"
                from_pretrained_kwargs["quantization_config"] = quant_config
                print("已设置回退的量化配置。")

            # 加载模型
            print("正在加载 GLM-4V 模型...")
            self.model = Glm4vForConditionalGeneration.from_pretrained(**from_pretrained_kwargs).eval()
            print(f"GLM-4V 模型 {model_name} 加载成功。")
            
        except Exception as e:
            # 如果加载失败，清理资源
            print(f"加载 GLM-4V 模型时出错: {e}")
            self.model = None
            self.processor = None
            self.current_model_name = None
            raise e

    def unload_model(self):
        """卸载当前加载的模型"""
        if self.model is not None:
            print(f"GLM-4V: 正在卸载模型 {self.current_model_name}。")
            # 删除模型和处理器
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.current_model_name = None
            # 如果使用CUDA，清空缓存
            if hasattr(self, 'device') and self.device == 'cuda':
                torch.cuda.empty_cache()
                print("已清空CUDA缓存。")
        
        # 取消定时器
        if self.unload_timer:
            self.unload_timer.cancel()
            self.unload_timer = None
            print("已取消定时卸载任务。")

    def generate(self, image, prompt, model_name, unload_policy):
        """生成图像描述"""
        print(f"开始生成任务，使用模型: {model_name}")
        
        # 如果有定时器在运行，先取消
        if self.unload_timer:
            self.unload_timer.cancel()
            self.unload_timer = None
            print("已取消之前的定时卸载任务。")

        try:
            # 加载指定的模型
            print(f"正在加载模型: {model_name}")
            self._load_model(model_name)
        except Exception as e:
            error_msg = f"加载模型时出错: {e}"
            print(error_msg)
            return (error_msg,)

        # 将tensor转换为PIL图像
        print("正在转换输入图像格式...")
        pil_image = self.tensor_to_pil(image)

        # 构建消息格式
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ],
            }
        ]
        print(f"已构建输入消息，提示词: {prompt}")
        
        try:
            # 确定输入设备
            device_map = self.model.hf_device_map
            if device_map and len(set(device_map.values())) > 1:
                input_device = "cpu"  # 如果模型分布在多个设备上，输入使用CPU
                print("模型分布在多个设备上，输入将使用CPU。")
            else:
                input_device = self.model.device  # 否则使用模型所在设备
                print(f"模型在单一设备上，输入将使用: {input_device}")

            # 应用聊天模板并处理输入
            print("正在处理输入数据...")
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(input_device)
            print("输入数据处理完成。")

            # 生成文本
            print("正在生成响应...")
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=8192, 
                do_sample=False
            )
            print("响应生成完成。")
            
            # 解码输出文本
            print("正在解码输出文本...")
            output_text = self.processor.decode(
                generated_ids[0][inputs["input_ids"].shape[1]:], 
                skip_special_tokens=True
            )
            print(f"解码完成，输出长度: {len(output_text)} 字符")
            
        except Exception as e:
            error_msg = f"生成过程中出错: {e}"
            print(error_msg)
            return (error_msg,)
        
        # 根据卸载策略处理模型
        if unload_policy == "Always":
            print("GLM-4V: 根据'Always'策略，生成完成后立即卸载模型。")
            self.unload_model()
        elif unload_policy != "Never":
            # 设置延时卸载
            delay_map = {
                "After 1 min": 60,
                "After 2 mins": 120,
                "After 5 mins": 300,
                "After 10 mins": 600
            }
            delay = delay_map.get(unload_policy)
            if delay:
                print(f"GLM-4V: 计划在 {delay} 秒后卸载模型。")
                self.unload_timer = threading.Timer(delay, self.unload_model)
                self.unload_timer.start()
        else:
            print("GLM-4V: 根据'Never'策略，模型将保持加载状态。")

        # 尝试提取答案标签中的内容
        match = re.search(r'.*<answer>(.*?)</answer>', output_text, re.DOTALL)
        if match:
            extracted_text = match.group(1).strip()
            print("成功提取到answer标签中的内容。")
            return (extracted_text,)
        else:
            print("未找到answer标签，返回完整输出。")
            return (output_text,)


class Glm4vBatchNode:
    """批量图片处理节点"""
    
    def __init__(self):
        # 共享Glm4vNode的模型实例以避免重复加载
        self.glm4v_node = Glm4vNode()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_folder_path": ("STRING", {  # 图片文件夹路径
                    "multiline": False,
                    "default": "C:/path/to/your/images",
                    "placeholder": "输入图片文件夹路径，例如: C:/Users/用户名/Pictures"
                }),
                "prompt": ("STRING", {  # 批量处理的提示词
                    "multiline": True,
                    "default": "describe this image,Describe in long sentence form, without using Markdown format."
                }),
                "model_name": (  # 模型选择
                    ["glmv4_4bit", "GLM-4.1V-9B-Thinking"],
                    {"default": "glmv4_4bit"}
                ),
                "image_extensions": ("STRING", {  # 支持的图片格式
                    "multiline": False,
                    "default": "jpg,jpeg,png,bmp,tiff,webp",
                    "placeholder": "支持的图片格式，用逗号分隔"
                }),
                "max_images": ("INT", {  # 最大处理图片数量
                    "default": 10,
                    "min": 1,
                    "max": 1000,
                    "step": 1
                }),
                "save_results": ("BOOLEAN", {  # 是否保存结果到文件
                    "default": True
                }),
                "output_format": (  # 输出格式选择
                    ["TXT", "JSON", "CSV"],
                    {"default": "TXT"}
                ),
                "unload_policy": (  # 模型卸载策略
                    ["Always", "Never", "After 1 min", "After 2 mins", "After 5 mins", "After 10 mins"],
                    {"default": "After 5 mins"}
                )
            },
        }

    RETURN_TYPES = ("STRING", "STRING",)  # 返回处理结果和统计信息
    RETURN_NAMES = ("batch_results", "statistics",)  # 返回值名称
    FUNCTION = "batch_generate"  # 执行函数名
    CATEGORY = "GLM4V"  # 节点分类

    def load_images_from_folder(self, folder_path, extensions, max_images):
        """从文件夹加载图片"""
        print(f"开始从文件夹加载图片: {folder_path}")
        
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"图片文件夹不存在: {folder_path}")
        
        # 支持的图片扩展名
        ext_list = [ext.strip().lower() for ext in extensions.split(',')]
        print(f"支持的图片格式: {ext_list}")
        
        # 搜索所有支持的图片文件
        image_files = []
        for ext in ext_list:
            # 搜索当前扩展名的文件（不区分大小写）
            pattern1 = os.path.join(folder_path, f"*.{ext}")
            pattern2 = os.path.join(folder_path, f"*.{ext.upper()}")
            image_files.extend(glob.glob(pattern1))
            image_files.extend(glob.glob(pattern2))
        
        # 去重并排序
        image_files = sorted(list(set(image_files)))
        
        # 限制图片数量
        if len(image_files) > max_images:
            print(f"找到 {len(image_files)} 张图片，限制处理前 {max_images} 张")
            image_files = image_files[:max_images]
        else:
            print(f"找到 {len(image_files)} 张图片，全部处理")
        
        return image_files

    def pil_to_tensor(self, pil_image):
        """将PIL图像转换为ComfyUI tensor格式"""
        try:
            # 确保图像是RGB格式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 将PIL图像转换为numpy数组
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            
            # 转换为torch tensor并添加batch维度 (1, H, W, C)
            tensor = torch.from_numpy(image_np).unsqueeze(0)
            
            return tensor
        except Exception as e:
            print(f"图像转换失败: {e}")
            raise e

    def save_results_to_file(self, results, folder_path, output_format):
        """保存结果到文件"""
        try:
            # 生成输出文件名（使用时间戳）
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            if output_format == "TXT":
                # 保存为文本文件
                output_file = os.path.join(folder_path, f"glm4v_results_{timestamp}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    for result in results:
                        f.write(f"文件: {result['filename']}\n")
                        f.write(f"描述: {result['description']}\n")
                        f.write(f"处理时间: {result['processing_time']:.2f}秒\n")
                        f.write("-" * 80 + "\n")
                        
            elif output_format == "JSON":
                # 保存为JSON文件
                import json
                output_file = os.path.join(folder_path, f"glm4v_results_{timestamp}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                    
            elif output_format == "CSV":
                # 保存为CSV文件
                import csv
                output_file = os.path.join(folder_path, f"glm4v_results_{timestamp}.csv")
                with open(output_file, 'w', newline='', encoding='utf-8-sig') as f:
                    fieldnames = ['filename', 'description', 'processing_time', 'status']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    for result in results:
                        writer.writerow(result)
            
            print(f"结果已保存到: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"保存结果文件时出错: {e}")
            return None

    def save_individual_descriptions(self, results, folder_path):
        """为每张图片保存单独的描述文件，文件名与图片相同"""
        try:
            saved_files = []  # 记录保存的文件列表
            
            # 为每个结果创建单独的txt文件
            for result in results:
                # 检查处理是否成功
                if result['status'] != 'success':
                    print(f"跳过失败的图片: {result['filename']}")
                    continue
                
                # 获取图片文件的基本名称（不包含扩展名）
                filename_without_ext = os.path.splitext(result['filename'])[0]
                
                # 创建txt文件路径
                txt_filename = f"{filename_without_ext}.txt"
                txt_filepath = os.path.join(folder_path, txt_filename)
                
                try:
                    # 写入描述内容到txt文件（只包含描述，不包含其他信息）
                    with open(txt_filepath, 'w', encoding='utf-8') as f:
                        f.write(result['description'])
                    
                    saved_files.append(txt_filepath)
                    print(f"已保存描述文件: {txt_filename}")
                    
                except Exception as e:
                    print(f"保存描述文件 {txt_filename} 时出错: {e}")
            
            print(f"总共保存了 {len(saved_files)} 个描述文件")
            return saved_files
            
        except Exception as e:
            print(f"保存单独描述文件时出错: {e}")
            return []

    def batch_generate(self, image_folder_path, prompt, model_name, image_extensions, 
                      max_images, save_results, output_format, unload_policy):
        """批量生成图片描述"""
        print(f"开始批量处理任务，文件夹: {image_folder_path}")
        print(f"使用模型: {model_name}, 最大图片数: {max_images}")
        
        results = []  # 存储所有结果
        success_count = 0  # 成功处理的图片数
        error_count = 0  # 处理失败的图片数
        total_time = 0  # 总处理时间
        
        try:
            # 加载图片文件列表
            image_files = self.load_images_from_folder(image_folder_path, image_extensions, max_images)
            
            if not image_files:
                return ("未找到任何图片文件", "处理统计: 0张图片被处理")
            
            print(f"开始处理 {len(image_files)} 张图片...")
            
            # 逐个处理图片
            for i, image_path in enumerate(image_files):
                print(f"\n处理第 {i+1}/{len(image_files)} 张图片: {os.path.basename(image_path)}")
                
                # 记录单个图片处理开始时间
                start_time = time.time()
                
                try:
                    # 加载并转换图片
                    print(f"正在加载图片: {image_path}")
                    pil_image = Image.open(image_path)
                    tensor_image = self.pil_to_tensor(pil_image)
                    
                    # 生成描述
                    description_result = self.glm4v_node.generate(
                        tensor_image, prompt, model_name, "Never"  # 批量处理时不自动卸载
                    )
                    
                    # 提取描述文本
                    description = description_result[0] if description_result else "生成失败"
                    
                    # 计算处理时间
                    processing_time = time.time() - start_time
                    total_time += processing_time
                    
                    # 保存结果
                    result = {
                        'filename': os.path.basename(image_path),
                        'full_path': image_path,
                        'description': description,
                        'processing_time': processing_time,
                        'status': 'success'
                    }
                    results.append(result)
                    success_count += 1
                    
                    print(f"处理成功，耗时: {processing_time:.2f}秒")
                    print(f"描述: {description[:100]}..." if len(description) > 100 else f"描述: {description}")
                    
                except Exception as e:
                    # 处理单个图片失败
                    error_msg = f"处理图片失败: {str(e)}"
                    print(error_msg)
                    
                    processing_time = time.time() - start_time
                    total_time += processing_time
                    
                    result = {
                        'filename': os.path.basename(image_path),
                        'full_path': image_path,
                        'description': error_msg,
                        'processing_time': processing_time,
                        'status': 'error'
                    }
                    results.append(result)
                    error_count += 1
            
            # 根据卸载策略处理模型
            if unload_policy == "Always":
                print("批量处理完成，立即卸载模型。")
                self.glm4v_node.unload_model()
            elif unload_policy != "Never":
                # 设置延时卸载
                delay_map = {
                    "After 1 min": 60,
                    "After 2 mins": 120,
                    "After 5 mins": 300,
                    "After 10 mins": 600
                }
                delay = delay_map.get(unload_policy)
                if delay:
                    print(f"批量处理完成，计划在 {delay} 秒后卸载模型。")
                    if self.glm4v_node.unload_timer:
                        self.glm4v_node.unload_timer.cancel()
                    self.glm4v_node.unload_timer = threading.Timer(delay, self.glm4v_node.unload_model)
                    self.glm4v_node.unload_timer.start()
            
            # 保存结果到文件（如果需要）
            output_file = None
            individual_files = []
            if save_results and results:
                # 保存传统格式的结果文件
                output_file = self.save_results_to_file(results, image_folder_path, output_format)
                # 为每张图片保存单独的描述文件（与图片同名）
                individual_files = self.save_individual_descriptions(results, image_folder_path)
            
            # 生成批量结果文本
            batch_results = []
            for result in results:
                batch_results.append(f"文件: {result['filename']}")
                batch_results.append(f"状态: {result['status']}")
                batch_results.append(f"描述: {result['description']}")
                batch_results.append(f"处理时间: {result['processing_time']:.2f}秒")
                batch_results.append("-" * 60)
            
            batch_results_text = "\n".join(batch_results)
            
            # 生成统计信息
            avg_time = total_time / len(image_files) if image_files else 0
            statistics = (
                f"批量处理统计信息:\n"
                f"总图片数: {len(image_files)}\n"
                f"成功处理: {success_count}\n"
                f"处理失败: {error_count}\n"
                f"总耗时: {total_time:.2f}秒\n"
                f"平均耗时: {avg_time:.2f}秒/张\n"
                f"使用模型: {model_name}\n"
            )
            
            if output_file:
                statistics += f"结果文件: {output_file}\n"
            if individual_files:
                statistics += f"单独描述文件: 已保存 {len(individual_files)} 个txt文件\n"
            
            print("\n" + statistics)
            
            return (batch_results_text, statistics)
            
        except Exception as e:
            error_msg = f"批量处理过程中出现错误: {str(e)}"
            print(error_msg)
            return (error_msg, f"处理失败: {error_msg}")


# 导出节点映射字典
# 注意: 名称应该是全局唯一的
NODE_CLASS_MAPPINGS = {
    "Glm4vNode": Glm4vNode,
    "Glm4vBatchNode": Glm4vBatchNode
}

# 节点的友好/人类可读标题映射字典
NODE_DISPLAY_NAME_MAPPINGS = {
    "Glm4vNode": "GLM-4V Generate",
    "Glm4vBatchNode": "GLM-4V Batch Generate"
}