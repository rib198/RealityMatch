
import torch

# المسار إلى نموذج عزيز
model_path = "aziz_model.pth"  # عدلي هذا لو الملف في مكان ثاني

# نحاول تحميل النموذج
try:
    model = torch.load(model_path, map_location='cpu')
    print("\n✅ تم تحميل النموذج بنجاح\n")

    # نعرض نوع المحتوى
    print("نوع المحتوى:", type(model))

    # إذا كان ديكشنري نعرض مفاتيحه
    if isinstance(model, dict):
        print("🔑 المفاتيح داخل النموذج:")
        for key in model.keys():
            print("-", key)
    else:
        # نعرض النموذج كامل
        print("محتوى النموذج:")
        print(model)

except Exception as e:
    print("❌ حصل خطأ أثناء تحميل النموذج:")
    print(e)
