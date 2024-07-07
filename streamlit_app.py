# coding: utf-8
import pandas as pd
import streamlit as st
import time
from ultralytics import YOLO
# from PIL import Image, ImageFont
import torch

from DetectUtils import *

# pytorch配置设备
# DEVICE_NAME = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'  # * nvidia cuda or mac m1
# DEVICE = torch.device(DEVICE_NAME)
DEVICE_NAME = 'cpu'  # * cpu only
DEVICE = torch.device("cpu")  # * cpu only
print(f'torch {torch.__version__}', f'; device: {DEVICE}')

# 初始化图片检测模型
model = None
results = None
org_img = None
location_list = None
cls_list = None
conf_list = None
id_list = None
seg_percent_list = None
draw_img = None
conf = 0.45
iou = 0.7
draw_seg = None
draw_box = None
mask_opts = None
mask_sel_opt = None
show_width = 610
show_height = 380
img_width = None
img_height = None


# 配置图片检测模型
def setup_models():
    global model, conf, iou

    with st.spinner("Loading YOLO-v8 pre-trained model, please wait..."):
        # 加载检测模型
        model = YOLO(model_path, task='segment')
        model(np.zeros((48, 48, 3)).astype(np.uint8), device=DEVICE)  # 预先加载推理模型


def do_detection(uploaded_file):
    global model, conf, iou, draw_seg, draw_box, mask_opts, mask_sel_opt, results, org_img, location_list, cls_list, conf_list, id_list, \
        seg_percent_list, draw_img, img_width, img_height, show_width, show_height

    img = Image.open(uploaded_file)
    org_img = img_cvread(img)

    # 目标检测
    t1 = time.time()
    results = model(img, conf=conf, iou=iou)[0]
    t2 = time.time()
    take_time_str = '{:.3f}s'.format(t2 - t1)

    location_list = results.boxes.xyxy.tolist()
    location_list = [list(map(int, e)) for e in location_list]
    cls_list = results.boxes.cls.tolist()
    cls_list = [int(i) for i in cls_list]
    conf_list = results.boxes.conf.tolist()
    conf_list = ['%.2f %%' % (each * 100) for each in conf_list]
    id_list = [i for i in range(len(location_list))]

    # 原始图像高和宽
    img_height, img_width = results.orig_shape
    total_px = img_height * img_width  # 图片总像素点
    seg_percent_list = []
    if results.masks is None:
        # 计算总分割面积百分比
        seg_area_px = '0 %'
    else:
        # 计算总分割面积百分比
        seg_area_px = cal_seg_percent(results.masks.xy, total_px)
        for each_seg in results.masks.xy:
            # print(each_seg)
            # 计算每个分割目标面积百分比
            seg_percent = cal_seg_percent([each_seg], total_px)
            seg_percent_list.append(seg_percent)

    # 目标数目
    target_nums = len(cls_list)

    # 准备标记数据
    target_list = [i for i in zip(location_list, cls_list, conf_list)]

    # 诊断结果判断
    res = names[-1]
    res_type_ind = -1
    res_ind = -1
    if 1 in cls_list:
        res = names[1]
        res_type_ind = 1
        res_ind = cls_list.index(1)
    elif 0 in cls_list:
        res = names[0]
        res_type_ind = 0
        res_ind = cls_list.index(0)

    with st.expander('Windows', expanded=True):
        # 设置目标选择下拉框
        st.subheader('Targets')
        choose_list = ['All']
        target_names = [names[id] + '_' + str(index) for index, id in enumerate(cls_list)]
        choose_list = choose_list + target_names

        # 分栏显示
        col1, col2 = st.columns(2)

        # 绘制窗口1图片
        with col1:
            target_sel_opt = st.selectbox('Select Targets', choose_list, index=0)
            target_ind = choose_list.index(target_sel_opt)

            show_ind = res_ind
            if target_ind > 0:
                show_ind = target_ind - 1
            kind_type_ind = cls_list[show_ind] if show_ind >= 0 else -1
            kind_res = names[kind_type_ind]
            area_percent = seg_percent_list[show_ind] if show_ind >= 0 else '<None>'
            ci = conf_list[show_ind] if show_ind >= 0 else '<None>'
            xmin, ymin, xmax, ymax = location_list[show_ind] if show_ind >= 0 else (
                '<None>', '<None>', '<None>', '<None>',)

            sc1, sc2 = st.columns(2)
            with sc1:
                st.write(f'xmin: {xmin}')
            with sc2:
                st.write(f'ymin: {ymin}')

            st.subheader('Window 1')
            # 自己绘制框与标签

            # 默认画标签方式
            # if target_ind == 0:
            #     # 选择了“All”（全部）
            #     draw_img = results.plot(boxes=draw_box, masks=draw_seg)
            # else:
            #     draw_img = results[show_ind].plot(boxes=draw_box, masks=draw_seg)
            # 自定义画标签方式
            draw_img = draw_rect_box_with_results(target_ind, target_list, org_img)

            st.image(draw_img, channels="BGR")

        # 绘制窗口2图片
        with col2:
            st.write(f'Kind Result: :{font_colors[kind_type_ind]}-background[{kind_res}]')
            sc1, sc2 = st.columns(2)
            with sc1:
                st.write(f'Area Percent: {area_percent}')
                st.write(f'xmax: {xmax}')
            with sc2:
                st.write(f'CI: {ci}')
                st.write(f'ymax: {ymax}')

            st.subheader('Window 2')
            if target_ind == 0:
                # 选择了“All”（全部）
                win2_img = draw_seg_mask(results, mask_opts.index(mask_sel_opt) == 0)
            else:
                win2_img = draw_seg_mask(results[show_ind], mask_opts.index(mask_sel_opt) == 0)
            st.image(win2_img, channels="BGR")

    with st.expander('Diagnosis results', expanded=True):
        st.subheader('Diagnosis results')
        st.write(f'Kind Result: :{font_colors[res_type_ind]}-background[{res}]')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f'Total numbers: {target_nums}')
        with col2:
            st.write(f'Total area percent: {seg_area_px}')
        with col3:
            st.write(f'Take time: {take_time_str}')
        # 多个结果显示在表格中
        new_cls_list = [names[-1 if i >= len(names) else i] for i in cls_list]
        df_results = pd.DataFrame([i for i in zip(id_list, new_cls_list, conf_list, location_list, seg_percent_list)],
                                  columns=['Target Id', 'Kind', 'CI', 'Coordinates', 'Area Percent'])
        st.table(df_results)
        st.write(f'*Comment: Coordinates value is described as `[xmin, ymin, xmax, ymax]`*')


# 渲染UI界面
def render_ui():
    global conf, iou, draw_seg, draw_box, mask_opts, mask_sel_opt

    st.title("Dr. Z.C.M.")
    st.header("Breast cancer intelligent detection, segmentation, and diagnosis.", divider='rainbow')
    st.subheader("""Select a file to upload, and get diagnosis results.""")

    uploaded_file = st.file_uploader("Choose a file", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", ],
                                     accept_multiple_files=False)

    with st.expander("CLICK HERE: Change the parameters"""):
        st.subheader("""Advanced Parameters""")
        conf = st.slider("Confidence Threshold", 0.0, 1.0, 0.45)
        iou = st.slider("Intersection-over-Union Threshold", 0.0, 1.0, 0.7)

        col1, col2 = st.columns(2)

        with col1:
            st.write('In Window 1:')
            draw_seg = st.toggle('Display segmentation results', True)
            draw_box = st.toggle('Display detection boxes and labels', True)
        with col2:
            st.write('In Window 2:')
            mask_opts = ['Mask', 'Original Segmentation Image']
            mask_sel_opt = st.radio('Display mask or original segmentation image',
                                    mask_opts,
                                    index=0, horizontal=True)

    if uploaded_file is not None:
        with st.expander("CLICK HERE: See the original image file"""):
            # 显示上传的图像
            st.subheader("Your Uploaded Image File:")
            col1, _ = st.columns(2)
            with col1:
                st.image(Image.open(uploaded_file))

        # 初始化图片检测模型
        setup_models()

        # 将上传数据送入模型，执行预测
        with st.spinner("Diagnosing with `YOLO-v8`, please wait..."):
            do_detection(uploaded_file)


# 主程序入口
if __name__ == "__main__":
    render_ui()
