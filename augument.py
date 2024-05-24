from PIL import Image
from torchvision import transforms

# 加载原始图片
img = Image.open("img/plane.jpg")

# 创建第一个图像转换流水线
transform1 = transforms.Compose([
    transforms.RandomCrop(233, padding=4),
    transforms.Pad(105)
])

# 创建第二个图像转换流水线
transform2 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0  )
])

# 应用第一个图像转换
transformed_img1 = transform1(img)

# 应用第二个图像转换
transformed_img2 = transform2(img)

# 保存第一个转换后的图片
transformed_img1.save("img/randomcrop_plane.jpg")

# 保存第二个转换后的图片
transformed_img2.save("img/randomhorizontalflip_plane.jpg")


