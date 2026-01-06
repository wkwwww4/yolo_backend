import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont

def batch_detect_videos(input_folder, output_folder="batch_results"):
    """
    批量检测文件夹中的所有视频文件，分别统计每个视频的人流
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有视频文件（包括HEVC格式）
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.MP4', '*.hevc', '*.h265']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_folder, ext)))
    
    # 如果没有找到视频，尝试直接列出所有文件
    if len(video_files) == 0:
        print("未找到标准视频文件，尝试列出所有文件...")
        all_files = os.listdir(input_folder)
        video_files = [os.path.join(input_folder, f) for f in all_files 
                      if any(f.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.hevc', '.h265'])]
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 加载模型
    model = YOLO("yolov8n.pt")
    
    # 存储每个视频的统计结果
    video_stats = {}
    
    # 批量处理每个视频
    for i, video_path in enumerate(video_files):
        video_name = os.path.basename(video_path)
        print(f"\n{'='*60}")
        print(f"正在处理视频 {i+1}/{len(video_files)}: {video_name}")
        print(f"{'='*60}")
        
        # 为每个视频创建单独的输出路径
        video_output_name = os.path.splitext(video_name)[0] + "_人流统计.mp4"
        video_output_path = os.path.join(output_folder, video_output_name)
        
        try:
            # 处理单个视频并获取统计结果
            total_people = process_single_video(model, video_path, video_output_path)
            
            # 记录该视频的统计结果
            video_stats[video_name] = total_people
            
            print(f"✓ 视频 {video_name} 分析完成！总人数: {total_people}")
            print(f"✓ 结果视频保存为: {video_output_name}")
            
        except Exception as e:
            print(f"✗ 处理视频 {video_name} 时出错: {str(e)}")
            video_stats[video_name] = f"处理失败: {str(e)}"
    
    # 输出批量处理总结
    print("\n" + "="*60)
    print("批量视频分析完成！各视频人流统计:")
    print("="*60)
    for video_name, people_count in video_stats.items():
        print(f"  📹 {video_name}: {people_count}")
    
    return video_stats

def check_video_format(video_path):
    """
    检查视频格式和编码信息
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "无法打开视频文件"
    
    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    cap.release()
    
    info = {
        'fps': fps,
        'width': width,
        'height': height,
        'total_frames': total_frames,
        'duration': total_frames / fps if fps > 0 else 0
    }
    
    return True, info

def process_single_video(model, input_video, output_video):
    """
    处理单个视频并返回总人数
    """
    # 首先检查视频是否可以打开
    success, video_info = check_video_format(input_video)
    if not success:
        raise Exception(f"视频格式不支持: {video_info}")
    
    print(f"视频信息: {video_info['width']}x{video_info['height']}, "
          f"{video_info['fps']:.1f} FPS, {video_info['total_frames']} 帧")
    
    # 用于记录所有出现过的行人ID
    all_people_ids = set()
    # 用于记录每个ID的颜色，确保同一ID颜色一致
    id_colors = {}
    
    # 打开视频文件
    cap = cv2.VideoCapture(input_video)
    fps = video_info['fps']
    width = video_info['width']
    height = video_info['height']
    total_frames = video_info['total_frames']
    
    # 设置输出视频（使用更兼容的编码）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'XVID' 如果mp4v不行
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    if not out.isOpened():
        # 尝试其他编码
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_video = output_video.replace('.mp4', '.avi')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    # 使用YOLO内置的ByteTrack跟踪器，逐帧处理
    results = model.track(
        source=input_video,
        classes=[0],  # 只检测行人
        conf=0.5,     # 置信度阈值
        tracker="bytetrack.yaml",
        stream=True,  # 逐帧处理
        persist=True
    )
    
    # 逐帧处理并统计人数
    frame_count = 0
    for result in results:
        frame_count += 1
        frame = result.orig_img.copy()
        
        # 获取当前帧的跟踪ID
        current_ids = []
        if result.boxes.id is not None:
            current_ids = result.boxes.id.cpu().numpy().astype(int)
            all_people_ids.update(current_ids)
            
            # 为每个检测到的人绘制边界框和ID
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            
            for i, (box, conf, track_id) in enumerate(zip(boxes, confs, current_ids)):
                x1, y1, x2, y2 = map(int, box)
                
                # 为每个ID分配固定颜色
                if track_id not in id_colors:
                    # 生成鲜艳的颜色
                    hue = (track_id * 50) % 180
                    color = tuple(map(int, cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]))
                    id_colors[track_id] = color
                
                color = id_colors[track_id]
                
                # 绘制边界框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制ID标签背景
                label = f"ID:{track_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
                
                # 绘制ID文本
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 绘制统计信息（使用英文避免字体问题）
        cv2.rectangle(frame, (10, 10), (300, 90), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (300, 90), (255, 255, 255), 2)
        
        cv2.putText(frame, f"Current: {len(current_ids)}", (20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Total: {len(all_people_ids)}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        cv2.putText(frame, f"Progress: {frame_count}/{total_frames}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 保存带统计信息的帧
        out.write(frame)
        
        # 每100帧打印一次进度
        if frame_count % 100 == 0:
            print(f"  已处理 {frame_count}/{total_frames} 帧，累计总人数: {len(all_people_ids)}")
    
    # 释放资源
    cap.release()
    out.release()
    
    return len(all_people_ids)

# 使用示例
if __name__ == "__main__":
    input_folder = "video"  # 替换为你的视频文件夹路径
    output_folder = "tracking_results"  # 结果保存文件夹
    
    # 执行批量处理
    results = batch_detect_videos(input_folder, output_folder)
    
    # 将结果保存到文本文件
    with open(os.path.join(output_folder, "人流统计报告.txt"), "w", encoding="utf-8") as f:
        f.write("各视频人流统计报告\n")
        f.write("=" * 50 + "\n")
        for video_name, people_count in results.items():
            f.write(f"{video_name}: {people_count} 人\n")
    
    print(f"\n详细报告已保存到: {os.path.join(output_folder, '人流统计报告.txt')}")