# **Elements of AI - Course Series**

**Building AI - Course**

**My AI Idea**

University of Helsinki, FI

**[Course page](https://buildingai.elementsofai.com/)**

# **Project Report**

# **BlindSight: AI-Powered Bio-Surveillance Solution**

**Summary:**
BlindSight is an advanced bio-surveillance system empowered by artificial intelligence, aimed at early detection and rapid response to emerging public health threats. By integrating sophisticated data analytics and machine learning algorithms, BlindSight provides actionable insights to healthcare professionals and security agencies, enhancing preparedness and response capabilities.

**Background:**
In an increasingly interconnected world, the threat of infectious diseases and bioterrorism demands proactive surveillance and intervention measures. Traditional surveillance methods often struggle to cope with the scale and complexity of modern health threats, leading to delays in detection and response. BlindSight addresses these challenges by leveraging AI to analyze diverse data sources and identify potential health threats in real-time. The urgency of this topic is underscored by the now stagnant COVID-19 pandemic and the ever-present risk of bioterrorism incidents.

**System Usage:**
BlindSight serves as a critical tool for public health agencies, medical institutions, and security organizations, providing continuous monitoring and alerting capabilities. Healthcare professionals, epidemiologists, and security analysts utilize BlindSight to analyze data from various sources, including healthcare records, laboratory results, environmental sensors, and global disease surveillance networks. The system operates seamlessly in diverse environments and situations, facilitating timely response and containment efforts in the face of emerging health threats.

# **Prototype System Implementation**

BlindSight's prototype system is built using Python programming language and leverages the scikit-learn library for implementing an anomaly detection algorithm. Specifically, the Isolation Forest algorithm is utilized due to its effectiveness in detecting anomalies in high-dimensional datasets. The implementation involves the following steps:

* **Data Preprocessing:** The system reads in data from a CSV file containing health-related data, such as patient records or laboratory results. The data is then preprocessed to handle missing values, normalize features, and ensure compatibility with the Isolation Forest algorithm.

* **Anomaly Detection:** The Isolation Forest algorithm is applied to the preprocessed data to identify anomalous instances that deviate significantly from the norm. Anomalies are flagged as potential indicators of emerging health threats and warrant further investigation by healthcare professionals and security analysts.

* **Alert Generation:** Upon detecting anomalies, the system generates alerts to notify relevant stakeholders, such as public health agencies or security organizations. Alerts may include information about the detected anomalies, their severity, and recommended actions for response and containment.

```
# Importing of the necessary libraries
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# This is a function for data preprocessing
def preprocess_data(data):

# Handle the missing values using mean imputation
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)
    
# Normalization features using standard scaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)
    
return data_scaled

# Function to detect anomalies using Isolation Forest algorithm
def detect_anomalies(data):

# Preprocessing of the data
data_processed = preprocess_data(data)
    
# Fitting of the Isolation Forest model
model = IsolationForest()
model.fit(data_processed)
    
# Predicting anomalies
nomalies = model.predict(data_processed)
    
return anomalies

# Example usage
def main():

# Loading the data from CSV file
data = pd.read_csv('health_data.csv')
    
# Detect anomalies using BlindSight AI
anomalies = detect_anomalies(data)
    
# Print a list of the detected anomalies
print("Detected anomalies:", anomalies)

# Execute main function
if __name__ == "__main__":
main()
```
# **System Explanation**

1. **Data Preprocessing:**
* **Missing Value Handling:** The data is preprocessed to handle missing values using mean imputation. This ensures that missing values are replaced with the mean value of the respective feature.
* **Feature Normalization:** Features are normalized using the StandardScaler to ensure uniform scale across different features. This step is crucial for improving the performance of the machine learning model.
  
2. **Anomaly Detection:**
* **Pipeline Construction:** A data preprocessing pipeline is constructed using scikit-learn's Pipeline class. This pipeline encapsulates the data preprocessing steps, ensuring consistency and ease of use.
* **Isolation Forest Algorithm:** An Isolation Forest model is utilized for anomaly detection. Isolation Forest is a tree-based anomaly detection algorithm that isolates outliers in the dataset.
  
3. **Example Usage:**
* **Loading of Data:** Health-related data is loaded from a CSV file ('health_data.csv'). This could include various types of health data such as patient records, laboratory results, or environmental sensor readings.
* **Anomaly Detection:** The BlindSight AI system is invoked to detect anomalies in the loaded dataset. Anomalies are instances that deviate significantly from the norm and may indicate potential health threats.

# **How it Works**

* **Data Loading:** The system loads health-related data from a CSV file into a pandas DataFrame.

* **Data Preprocessing:** The data undergoes preprocessing, including handling missing values and feature normalization, to ensure it is suitable for input into the machine learning model.

* **Anomaly Detection:** An Isolation Forest model is trained on the preprocessed data to detect anomalies. Anomalies are instances that are isolated from the majority of the data points, indicating potential health threats.

* **Alert Generation:** Detected anomalies are flagged as potential health threats, and appropriate actions, such as alerting relevant stakeholders or triggering response protocols, can be initiated based on the severity of the anomalies.

* **Data Sources and AI Methods:** BlindSight relies on a diverse array of data sources, including structured and unstructured data from healthcare systems, environmental sensors, social media, and global disease surveillance networks. AI techniques such as machine learning, anomaly detection algorithms (e.g., Isolation Forest), and natural language processing are employed to analyze and interpret these data sources, enabling the early detection of potential health threats.

# **Challenges**

BlindSight faces several challenges in its implementation and deployment.

* **Data Integration:** Ensuring seamless integration of data from disparate sources while maintaining data quality and integrity.

* **Algorithmic Accuracy:** Continuous refinement and validation of AI models to improve accuracy and reduce false positives/negatives.

* **Ethical Considerations:** Addressing privacy concerns and ensuring ethical use of sensitive health data in compliance with regulations and standards.

# **What's Next?**
BlindSight has the potential to evolve into a comprehensive global bio-surveillance network, incorporating advanced features such as real-time genomic sequencing, mobile health monitoring, and predictive analytics. Continued collaboration with international partners, ongoing research and development efforts, and investment in technology infrastructure will be key to realizing BlindSight's full potential in safeguarding public health and national security.

# **Acknowledgments**
BlindSight draws inspiration from existing bio-surveillance initiatives such as HealthMap, ProMED, and the Global Health Security Agenda (GHSA). Special thanks to the healthcare professionals, researchers, and technology partners working tirelessly to advance the field of bio-surveillance and protect global health security. 

# **References**

**For the programming part:**

* [1] [Python: The Python Software Foundation. (n.d.);](https://www.python.org/)
  
* [2] [scikit-learn: Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830;](https://scikit-learn.org/)
  
* [3] [Isolation Forest algorithm: Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In Proceedings of the 2008 Eighth IEEE International Conference on Data Mining (pp. 413-422). IEEE. doi:10.1109/ICDM.2008.17.](http://www.lamda.nju.edu.cn/publication/icdm08b.pdf)

**Everything else:**

* [1] [ProMED;](https://promedmail.org/)

* [2] [HealthMap;](https://www.healthmap.org/)

* [3] [World Health Organization (WHO);](https://www.who.int/)

* [4] [National Institutes of Health (NIH);](https://www.nih.gov/)

* [5] [Global Health Security Agenda (GHSA);](https://www.ghsagenda.org/)

* [6] [Centers for Disease Control and Prevention (CDC);](https://www.cdc.gov/)

* [7] [Johns Hopkins University - Center for Health Security;](https://www.centerforhealthsecurity.org/)

* [8] [European Centre for Disease Prevention and Control (ECDC).](https://www.ecdc.europa.eu/)


