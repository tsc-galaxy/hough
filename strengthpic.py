from PIL import Image
from PIL import ImageEnhance

img = Image.open('huiduhua.jpg')
img.show()

# 对比度增强
enh_con = ImageEnhance.Contrast(img)
contrast = 1.5
img_contrasted = enh_con.enhance(contrast)
img_contrasted.show()
img_contrasted.save("duibidu.jpg")