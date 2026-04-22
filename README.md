#  Lab 32 — Setting Up and Using a Cloud-Based Jupyter Environment

A comprehensive guide for getting started with **Google Colab** and cloud-based data science workflows. This lab walks you through setting up a Jupyter notebook in the cloud, installing packages, working with datasets, training machine learning models, and saving your work to Google Drive.

**Course:** Data Science — Week 7  
**Topic:** Cloud Computing for Data Science  
**Platform:** Google Colab (runs entirely in your browser, no installation needed!)

---

##  Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [What You Will Learn](#what-you-will-learn)
- [Lab Steps](#lab-steps)
- [Usage Instructions](#usage-instructions)
- [Technologies Used](#technologies-used)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

---

##  Overview

This lab demonstrates the complete workflow of data science in a cloud environment using **Google Colab**. You'll go from zero to a trained machine learning model without installing anything on your local machine.

**Key highlights:**
-  Launch and configure a Jupyter environment in Google Colab
-  Install and import Python libraries (pip package management)
-  Load real-world datasets (Iris flower dataset from scikit-learn)
-  Explore and visualize data with pandas and seaborn
-  Train a machine learning model (Random Forest Classifier)
-  Evaluate model performance and generate reports
-  Save trained models to Google Drive for persistence
-  Load and reuse trained models for new predictions

---

##  Prerequisites

- **Google Account** — required to use Google Colab
- **Google Drive** — to save and retrieve trained models
- **Web Browser** — Chrome, Firefox, Safari, or Edge (any modern browser)
- **No local installation required!** Everything runs in the cloud

---

##  What You Will Learn

### 1. **Cloud Jupyter Basics**
   - What Google Colab is and how it compares to local Jupyter
   - Running code cells and markdown cells
   - Using magic commands (`!`, `%`)

### 2. **Package Management**
   - Installing Python libraries with `!pip install`
   - Importing and managing dependencies
   - Handling pre-installed vs. additional packages

### 3. **Data Handling**
   - Loading datasets from scikit-learn
   - Exploring data with pandas (shape, describe, value_counts)
   - Understanding data structure and types

### 4. **Data Visualization**
   - Creating exploratory plots with matplotlib and seaborn
   - Generating pair plots for multivariate analysis
   - Best practices for data visualization

### 5. **Machine Learning Workflow**
   - Splitting data into train/test sets
   - Training a Random Forest Classifier
   - Making predictions on new data
   - Evaluating model performance (accuracy, classification reports)

### 6. **Model Persistence**
   - Saving trained models using joblib
   - Mounting and accessing Google Drive from Colab
   - Loading pre-trained models for inference

---

##  Lab Steps

### Step 1: Launch Google Colab
- Open [colab.research.google.com](https://colab.research.google.com)
- Create a new notebook or upload this `.ipynb` file
- No setup required — start coding immediately!

### Step 2: Authentication & Google Drive
- Mount Google Drive to save outputs and models
- Create a folder (`DS_Lab_Week7`) for organizing your work
- Verify read/write permissions

### Step 3: Import Essential Libraries
Libraries pre-installed in Colab include:
- `numpy` — numerical computing
- `pandas` — data manipulation
- `scikit-learn` — machine learning
- `matplotlib` & `seaborn` — visualization
- `tensorflow` — deep learning (optional for this lab)

### Step 4: Install Additional Packages
```python
!pip install -q yellowbrick  # Optional: advanced visualization
```

### Step 5: Load the Iris Dataset
```python
from sklearn.datasets import load_iris
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target_names[iris.target]
```

### Step 6: Exploratory Data Analysis (EDA)
```python
df.describe()
df['species'].value_counts()
sns.pairplot(df, hue='species')
```

### Step 7: Train a Machine Learning Model
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop('species', axis=1)
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f" Model Accuracy: {accuracy * 100:.2f}%")
```

### Step 8: Save the Model to Google Drive
```python
import joblib
import os

save_folder = '/content/drive/MyDrive/DS_Lab_Week7'
os.makedirs(save_folder, exist_ok=True)
joblib.dump(model, f'{save_folder}/iris_model.pkl')
```

### Step 9: Load and Reuse the Model
```python
loaded_model = joblib.load(f'{save_folder}/iris_model.pkl')
new_prediction = loaded_model.predict([[5.1, 3.5, 1.4, 0.2]])
```

---

##  Usage Instructions

### Opening the Notebook in Google Colab

**Option 1: Upload this file directly**
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click "File" → "Upload notebook"
3. Select `Lab_Cloud_Jupyter_Setup.ipynb`
4. Run cells from top to bottom (Shift + Enter)

**Option 2: Clone from GitHub and open in Colab**
```bash
# Copy this URL into your browser, replacing with your GitHub repo URL:
https://colab.research.google.com/github/YOUR_USERNAME/YOUR_REPO/blob/main/Lab_Cloud_Jupyter_Setup.ipynb
```

### Running the Notebook
- **Run a single cell:** Click the play button or press `Shift + Enter`
- **Run all cells:** Click "Runtime" → "Run all"
- **Clear output:** Click "Runtime" → "Clear all outputs"
- **Interrupt execution:** Click "Runtime" → "Interrupt execution"

### Expected Output
```
Flowers per species:
setosa         50
virginica      50
versicolor    50
Name: species, dtype: int64

 Model Accuracy: 100.00%
```

---

##  Technologies Used

| Technology | Purpose | Version |
|-----------|---------|---------|
| **Python** | Programming language | 3.13.9 |
| **Google Colab** | Cloud Jupyter environment | Latest |
| **pandas** | Data manipulation | Pre-installed |
| **scikit-learn** | Machine learning | Pre-installed |
| **matplotlib** | Plotting library | Pre-installed |
| **seaborn** | Statistical visualization | Pre-installed |
| **numpy** | Numerical computing | Pre-installed |
| **joblib** | Model serialization | Pre-installed |
| **yellowbrick** | ML visualization (optional) | Latest |

---

##  File Structure

```
.
├── Lab_Cloud_Jupyter_Setup.ipynb    # Main Jupyter notebook
├── README.md                         # This file
├── requirements.txt                  # (Optional) Python dependencies
└── models/                           # (Optional) Directory for saved models
    └── iris_model.pkl               # Example saved model
```

### Key Files Explained

- **`Lab_Cloud_Jupyter_Setup.ipynb`** — Complete walkthrough with code, explanations, and outputs. Start here!
- **`README.md`** — Documentation and setup guide (you're reading it!)
- **`requirements.txt`** — List of Python packages needed (for reference/local setup)

---

##  Troubleshooting

### **Issue: "ModuleNotFoundError: No module named 'X'"**
**Solution:**
```python
!pip install package_name
```
Then re-run your code cell.

---

### **Issue: "Permission denied" when saving to Google Drive**
**Solution:**
1. Run the authentication cell: `from google.colab import drive; drive.mount('/content/drive')`
2. Verify you've granted permissions in the popup
3. Check that the directory exists before saving

---

### **Issue: Model file not found when loading**
**Solution:**
1. Verify the file path is correct: `print(os.path.exists(save_path))`
2. Check your Google Drive for the file manually
3. Ensure you're using the full `/content/drive/MyDrive/...` path

---

### **Issue: Kernel crashes or times out**
**Solution:**
1. Click "Runtime" → "Change runtime type" → Select "GPU" for faster execution
2. Break large operations into smaller cells
3. Restart the kernel: "Runtime" → "Restart runtime"

---

##  Resources

### Official Documentation
- [Google Colab Official Guide](https://colab.research.google.com/notebooks/intro.ipynb)
- [scikit-learn Documentation](https://scikit-learn.org)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [seaborn Gallery](https://seaborn.pydata.org/examples.html)

### Related Learning Materials
- [Random Forest Classifier Explained](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Iris Dataset Background](https://en.wikipedia.org/wiki/Iris_flower_data_set)
- [Train/Test Split Best Practices](https://scikit-learn.org/stable/modules/cross_validation.html)

### Community & Support
- [Stack Overflow `google-colab` tag](https://stackoverflow.com/questions/tagged/google-colab)
- [Colab GitHub Issues](https://github.com/googlecolab/colabtools)
- [kaggle.com](https://kaggle.com) — Datasets and community notebooks

---

##  Learning Outcomes

After completing this lab, you will be able to:

 Launch and configure cloud-based Jupyter notebooks  
 Install and manage Python packages in cloud environments  
 Load, explore, and visualize real datasets  
 Build and train machine learning models  
 Evaluate model performance quantitatively  
 Persist and reload trained models for production use  
 Collaborate using cloud infrastructure (Google Drive)  

---

##  License

This lab material is provided for educational purposes. Feel free to use, modify, and share for learning.

---

##  Questions & Feedback

If you encounter issues or have suggestions for improvement:
1. Check the **Troubleshooting** section above
2. Review official documentation links
3. Post in course forums or GitHub Issues

---

**Happy coding!  Welcome to cloud-based data science!**
