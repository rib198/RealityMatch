
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

print("🚀 بدء تحميل البيانات...")
X = np.load("X.npy")
y = np.load("y.npy")
print("✅ تم تحميل البيانات:", X.shape, y.shape)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

class AzizNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AzizNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

input_dim = X.shape[1]
hidden_dim = 64
output_dim = len(torch.unique(y_tensor))

model = AzizNet(input_dim, hidden_dim, output_dim)
print("✅ النموذج: AzizNet", input_dim, "→", hidden_dim, "→", output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print("📊 بدء التدريب مع الحفظ بعد كل Epoch...")

for epoch in range(10):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"📉 Epoch {epoch+1}/10 | Loss: {total_loss:.4f}")
    
    # ✅ حفظ النموذج بعد كل دورة
    torch.save(model.state_dict(), f"aziz_model_epoch{epoch+1}.pth")
    print(f"💾 تم الحفظ: aziz_model_epoch{epoch+1}.pth")

# حفظ نسخة نهائية
torch.save(model.state_dict(), "aziz_model.pth")
print("✅ النموذج النهائي محفوظ في aziz_model.pth")
