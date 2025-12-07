import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ตั้งค่า random seed เพื่อให้ผลลัพธ์ซ้ำได้
np.random.seed(42)

print("=" * 80)
print("Bayes Decision Theory - 2D Multivariate Normal Distribution")
print("=" * 80)

# ========== 1) Data Representation (Slide 19) ==========
print("\n[Step 1] Data Representation")
print("-" * 80)

# กำหนดพารามิเตอร์ตาม Slide 13
mean_A = np.array([0.0, 0.0])      # Label A: mean [0.0, 0.0]
mean_B = np.array([3.2, 0.0])      # Label B: mean [3.2, 0.0]

cov_A = np.array([[0.10, 0.00],    # Label A: covariance
                  [0.00, 0.75]])
cov_B = np.array([[0.75, 0.00],    # Label B: covariance
                  [0.00, 0.10]])

n_samples = 250  # 250 ตัวอย่างต่อ class

print(f"Label A: mean = {mean_A}, covariance = \n{cov_A}")
print(f"Label B: mean = {mean_B}, covariance = \n{cov_B}")
print(f"Samples per class: {n_samples}")

# สร้างข้อมูล
data_A = np.random.multivariate_normal(mean_A, cov_A, size=n_samples)
data_B = np.random.multivariate_normal(mean_B, cov_B, size=n_samples)

# รวมข้อมูล
X = np.vstack([data_A, data_B])
y = np.array([0] * n_samples + [1] * n_samples)  # 0 = Class A, 1 = Class B

print(f"Total samples: {len(X)}")

# ========== 2) Estimate Parameters (Slide 20) ==========
print("\n[Step 2] Estimate Parameters")
print("-" * 80)

# คำนวณ mean และ covariance จากข้อมูล (ในกรณีนี้เรารู้แล้ว)
mu_0 = mean_A
mu_1 = mean_B
Sigma_0 = cov_A
Sigma_1 = cov_B

print(f"μ₀ (Class A mean): {mu_0}")
print(f"μ₁ (Class B mean): {mu_1}")
print(f"Σ₀ (Class A covariance):\n{Sigma_0}")
print(f"Σ₁ (Class B covariance):\n{Sigma_1}")

# ========== 3) Calculate Class Priors (Slide 22) ==========
print("\n[Step 3] Calculate Class Priors")
print("-" * 80)

prior_0 = 0.5  # P(Class = 0)
prior_1 = 0.5  # P(Class = 1)

print(f"P(Class = 0) = {prior_0}")
print(f"P(Class = 1) = {prior_1}")

# ========== 4) Bayes' Theorem (Slide 23) ==========
print("\n[Step 4] Bayes' Theorem")
print("-" * 80)
print("P(Class|X) = P(X|Class) · P(Class) / P(X)")

# ========== 5) Calculate Class Conditional Probabilities (Slide 25) ==========
print("\n[Step 5] Calculate Class Conditional Probabilities")
print("-" * 80)

def multivariate_gaussian(x, mu, Sigma):
    """
    คำนวณ probability density function ของ Multivariate Gaussian
    P(X|Class) = 1/√((2π)^d·|Σ|) · exp(-0.5(X-μ)ᵀΣ⁻¹(X-μ))
    """
    d = len(mu)
    diff = x - mu
    det_Sigma = np.linalg.det(Sigma)
    inv_Sigma = np.linalg.inv(Sigma)
    
    coeff = 1.0 / np.sqrt((2 * np.pi)**d * det_Sigma)
    exponent = -0.5 * np.dot(np.dot(diff.T, inv_Sigma), diff)
    
    return coeff * np.exp(exponent)

print("P(X|Class = 0) = 1/√((2π)²·|Σ₀|) · exp(-0.5(X-μ₀)ᵀΣ₀⁻¹(X-μ₀))")
print("P(X|Class = 1) = 1/√((2π)²·|Σ₁|) · exp(-0.5(X-μ₁)ᵀΣ₁⁻¹(X-μ₁))")

# ========== 6) Decision Rule (Slide 26-31) ==========
print("\n[Step 6] Decision Rule")
print("-" * 80)

def bayes_classifier(x, mu_0, mu_1, Sigma_0, Sigma_1, prior_0, prior_1):
    """
    Bayes Classifier โดยใช้ Discriminant Function
    
    Discriminant = P(Class=0|X) / P(Class=1|X)
                 = [P(X|Class=0)·P(Class=0)] / [P(X|Class=1)·P(Class=1)]
    
    ถ้า Discriminant > 1 → Class 0
    ถ้า Discriminant < 1 → Class 1
    """
    # คำนวณ likelihood
    likelihood_0 = multivariate_gaussian(x, mu_0, Sigma_0)
    likelihood_1 = multivariate_gaussian(x, mu_1, Sigma_1)
    
    # คำนวณ posterior
    posterior_0 = likelihood_0 * prior_0
    posterior_1 = likelihood_1 * prior_1
    
    # Decision: เลือก class ที่มี posterior สูงกว่า
    return 0 if posterior_0 > posterior_1 else 1

print("Discriminant Function = P(Class=0|X) / P(Class=1|X)")
print("If Discriminant > 1 → Classify as Class 0")
print("If Discriminant < 1 → Classify as Class 1")

# ทำนาย Class
print("\n[Step 7] Classification")
print("-" * 80)

predictions = np.array([bayes_classifier(x, mu_0, mu_1, Sigma_0, Sigma_1, prior_0, prior_1) 
                        for x in X])

# คำนวณความแม่นยำ
accuracy = np.mean(predictions == y) * 100
correct = np.sum(predictions == y)

print(f"Accuracy: {accuracy:.2f}%")
print(f"Correct predictions: {correct}/{len(y)}")

# Confusion Matrix
TP = np.sum((predictions == 1) & (y == 1))
TN = np.sum((predictions == 0) & (y == 0))
FP = np.sum((predictions == 1) & (y == 0))
FN = np.sum((predictions == 0) & (y == 1))

print(f"\nConfusion Matrix:")
print(f"                Predicted A    Predicted B")
print(f"  Actual A         {TN:3d}            {FP:3d}")
print(f"  Actual B         {FN:3d}            {TP:3d}")

# ========== Visualization ==========
print("\n[Step 8] Visualization")
print("-" * 80)

# สร้าง mesh grid สำหรับ decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 2, X[:, 1].max() + 2
h = 0.05
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# ทำนาย class สำหรับทุกจุดใน grid
Z = np.array([bayes_classifier(np.array([x, y]), mu_0, mu_1, Sigma_0, Sigma_1, prior_0, prior_1)
              for x, y in zip(xx.ravel(), yy.ravel())])
Z = Z.reshape(xx.shape)

# สร้างกราฟ 3 แบบในรูปเดียว (ตามภาพที่ส่งมา)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ---------- กราฟที่ 1: Classwork p.18 (100D → 2D view) ----------
ax1 = axes[0]

# Plot data only (ไม่มี decision boundary)
ax1.scatter(data_A[:, 0], data_A[:, 1], c='red', marker='o', 
           s=30, alpha=0.6, label='Class A')
ax1.scatter(data_B[:, 0], data_B[:, 1], c='blue', marker='o', 
           s=30, alpha=0.6, label='Class B')

ax1.set_xlabel('Feature 1', fontsize=11)
ax1.set_ylabel('x2', fontsize=11)
ax1.set_title('Classwork p.18 (100D → 2D view)', fontsize=12, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', adjustable='box')

# ---------- กราฟที่ 2: Raw 2D data (p.19) ----------
ax2 = axes[1]

# Plot data only
ax2.scatter(data_A[:, 0], data_A[:, 1], c='red', marker='o', 
           s=30, alpha=0.6, label='Class 0')
ax2.scatter(data_B[:, 0], data_B[:, 1], c='blue', marker='o', 
           s=30, alpha=0.6, label='Class 1')

ax2.set_xlabel('x1', fontsize=11)
ax2.set_ylabel('x2', fontsize=11)
ax2.set_title('Raw 2D data (p.19)', fontsize=12, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal', adjustable='box')

# ---------- กราฟที่ 3: Bayes decision boundary (p.26-31) ----------
ax3 = axes[2]

# Decision regions with pink/blue background
colors = ['#FFB6C1', '#ADD8E6']  # Light pink and light blue
cmap = ListedColormap(colors)
ax3.contourf(xx, yy, Z, alpha=0.5, cmap=cmap, levels=[0, 0.5, 1])

# Decision boundary (เส้นดำหนา)
ax3.contour(xx, yy, Z, colors='black', linewidths=3, levels=[0.5])

# Plot data
ax3.scatter(data_A[:, 0], data_A[:, 1], c='red', marker='o', 
           s=30, alpha=0.7, label='Class 0')
ax3.scatter(data_B[:, 0], data_B[:, 1], c='blue', marker='o', 
           s=30, alpha=0.7, label='Class 1')

ax3.set_xlabel('x1', fontsize=11)
ax3.set_ylabel('x2', fontsize=11)
ax3.set_title('Bayes decision boundary (p.26-31)', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left', fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.savefig('bayes_classification_3plots.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: bayes_classification_3plots.png")

plt.show()

print("\n" + "=" * 80)
print("Bayes Decision Theory - Completed!")
print("=" * 80)