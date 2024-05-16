import torch
from torchvision import transforms
from mslf import MSLF  # 请替换为你实际的模型类
from dataset import BSDS_Dataset  # 请替换为你实际的数据集类
from PIL import Image


def load_model(model_path):
    model = MSLF()  # 请替换为你实际的模型类
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def test_image(model, image_path):
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    image = Image.open(image_path).convert('L')  # 以灰度模式打开图像
    image = transform(image)
    image = image.unsqueeze(0)  # 添加批次维度

    # 模型推理
    with torch.no_grad():
        output = model(image)

    # 处理输出（这里假设输出是概率图，可以根据实际情况调整）
    predicted_edges = output.squeeze().cpu().numpy()  # 去除批次维度，并转换为 NumPy 数组

    return predicted_edges


if __name__ == "__main__":
    model_path = "E:/Program/project/深度学习/edge/mslf1.pth"  # 请替换为你的模型路径
    image_path = "E:/Program/project/深度学习/edge/002.jpg"  # 请替换为你的测试图像路径

    # 加载模型
    model = load_model(model_path)

    # 进行测试
    predicted_edges = test_image(model, image_path)

    # 在这里你可以根据需要进行后续处理或可视化，比如保存结果等
