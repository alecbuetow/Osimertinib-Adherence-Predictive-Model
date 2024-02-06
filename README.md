# Humana Case Competition - Designing a Model to Predict Patient Drug Adherence and Providing a Plan to Address Non-Adherence

Predictive analytics were performed via a neural network in order to identify patients at risk of prematurely ending their Osimertinib treatment. An AUC of 0.93 was achieved on validation data through focused feature selection. This project was undertaken as part of the Humana-Mays Healthcare Analytics Case Competition, where our team proceeded to the semifinals, placing in the top 50 of 150 teams.

# Key Takeaways

- **Neural Network:** Achieved an AUC of 0.93 on Validation Set
- **Data Engineering:** Many observations linked to one outcome by aggregation
- **Business Report:** Actionable insights written into a 44-page report for business and subject-matter experts
  
# Overview

Osimertinib is a targeted therapy designed to _restrain_ uncontrollably dividing cells (as opposed to eliminating them) and is primarily used for non-small cell lung cancer (NSCLC). It serves as a maintenance medication prescribed to patients post-surgery, with the aim of reducing cancer recurrence. As it requires indefinite adherence to prevent tumor growth, patients who experience adverse side effects often forego treatment, ultimately losing faith that their health while on Osimertinib is any better than cancer progression. Identifying these patients who are more susceptible to prematurely ending their therapy is imperative, both to promote public health and to prevent increased costs of treatment down the line.

# Model & Data Processing

The neural network used in pursuit of this task considered the following features: physician and pharmacy visits, both in the forms of claims data, in addition to patient demographics. The many-to-one nature of this problem, with many claims corresponding to one patient and therefore their singular outcome (adherent or not) was addressed by aggregating visits as a count of reported diagnoses. Missing demographic data was imputed with the median or mode as appropriate, for example age and race, respectively. Missing claims data was imputed with zero, as health records are highly accurate and supposedly missing values made contextual sense to be zero in reality, for example the out of pocket cost of a given prescription.

# Recommendations Report

All analyses on the data, insights from the model and relevant visuals were organized into a 44 page recommendations report for Humana. Actionable steps to increase adherence were provided, in addition to key performance indicators which would measure the effects of these steps. Detailed plans, costs and projected benefits were organized into a narrative for ease of consumption by both industry and academic professionals. 

# Environment and Dependencies
This project was developed using Python version 3.7. The following Python packages and libraries were utilized for data analysis and statistical modeling:

- `pandas`
- `scikit-learn`
- `numpy`
- `matplotlib`
- `seaborn`
- `tensorflow`



