
prompt1  ='''
<Background>  
You are an expert in time series data analysis, capable of analyzing line plots, time-frequency images, and heatmaps. I will provide line plots, time-frequency images, and heatmaps of multiple samples under the same label.  

Your task is to: accurately extract all feature events from the three types of plots, and analyze whether each event is a typical or atypical feature based on the heatmap. A feature event refers to the specific manifestation in the line plot and time-frequency image of the colored (red or blue) time-frequency regions in the heatmap:  
- In the line plot, feature events may manifest as peaks, valleys, trend changes, jitter, shape changes, or other characteristics.  
- In the time-frequency image, feature events may manifest as high-energy regions in high/low frequencies (e.g., high-energy regions occupying adjacent or discrete frequency components within a certain time region).  
- In the heatmap, determine whether an event is a typical feature (red) or atypical feature (blue) based on color.  

The following is key background information for these three types of plots:  
- **Line plots**: The line plot of each sample can provide valuable classification features such as shape, rate of change, trend, peaks, and valleys. Since the numerical range of the Y-axis varies by sample, an auxiliary line y=0 is plotted to facilitate measuring the degree of deviation of data points from y=0, helping you analyze the magnitude relationship of peaks/valleys.  
- **Time-frequency images**: High-energy regions in time-frequency images correspond to peaks or valleys in line plots that deviate from y=0 — the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. Time-frequency images also provide rich frequency information, enabling the decomposition of signals into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as high-energy regions distributed across wider frequency components; the more intense the jitter, the more frequency components the high-energy regions may cover, thereby reflecting the dispersion of frequency distribution.  
- **Heatmaps**: Generated based on time-frequency images, heatmaps reflect the contribution of energy in different time regions and frequency components to classification. Red regions indicate that the energy of the current time period and frequency component is a typical feature of the corresponding label; blue regions indicate atypical features. The role of heatmaps is to directly associate the energy information of time-frequency images with classification labels, helping determine which feature events are crucial for classification.  

<Analysis Methods and Rules>  
Red and blue regions in the heatmap may be distributed in different time regions or different frequency components of the same time region. Therefore, the heatmap can not only reflect which time regions' events are typical/atypical features but also which frequency components of the same event are typical/atypical features. When summarizing the mapping relationship between features and heatmap colors, the following requirements must be strictly met:  
- For key features such as peaks/valleys, observe whether there is fluctuation in the plot. If there is fluctuation, you should **ignore the color mark of the bottom 0Hz frequency row in the heatmap** and focus on the color of the adjacent upper frequency component, which represents the true contribution of the feature. 0Hz represents a steady signal component and cannot reflect the contribution of fluctuating signals.  
- After identifying the mapping relationship between events and colors, describe the events by combining the line plot and time-frequency image. Examples will be provided below, and you must strictly follow the analysis ideas of the examples when analyzing similar or other events:  
  - If an oscillating peak is blue in the frequency rows above 0Hz in the heatmap (ignoring the color mark of the 0Hz row), it indicates that the amplitude of the fluctuation is an inhibitory feature; if the adjacent lower 0Hz frequency row shows a different color mark, record it in detail.  
  - If a baseline with a Y-axis numerical range of [X-Y] is marked blue in the corresponding region of the heatmap, while a baseline with a Y-axis numerical range of [A-B] is marked red in the corresponding region of the heatmap, it indicates that the Y-axis range [A-B] is the ideal numerical range for the baseline of the label, and a baseline deviating from this range will be regarded as an inhibitory feature.  

<Task>  
Please think according to the following steps:  
Step 1: Analyze the line plot, time-frequency image, and heatmap of each sample under the target label, and extract as much valuable information as possible; summarize all features appearing in each sample, their manifestations (numerical range, amplitude range, shape, trend, volatility, etc.), and the heatmap colors corresponding to different manifestations of these features. If necessary, divide and describe the frequency components (e.g., the 0 Hz row, the adjacent upper frequency row, the medium-frequency region, the high-frequency region, etc.)  
Step 2: Summarize the pattern of each sample, i.e., the combination relationship between the features analyzed in Step 1 and their different manifestations. Analyze which features each sample consists of, what the manifestations of these features are, and what color each corresponds to. If necessary, divide the sequence into meaningful segments (e.g., early, middle, late stages). (Example: The pattern of Sample X is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the late stage, where the early baseline is marked red in the heatmap and the late stable amplitude is marked blue in the heatmap);  
Step 3: Inductively analyze the results of each reference sample and summarize them into all feature lists and pattern features of the label.  

<Output Requirements>  
You must use JSON format to conduct a detailed analysis of the label to be summarized. The summary content must be no less than 400 words, covering valuable information such as trends, periodicity, baselines, peaks, valleys, stability, shape, rate of change, and the mapping relationship between features and heatmaps under different frequency components. Ensure that the output content meets all requirements and analysis rules.  

You must return the result in JSON format, and the format must strictly follow:  
{  
   "feature": ["List all features reflected by the samples, their occurrence timing, manifestations, and corresponding heatmap colors one by one. Features refer to baselines/peaks/valleys/others; manifestations refer to specific numerical values/stability/shape/others of these features. Different manifestations of the same feature under the same label (e.g., different numerical ranges/amplitudes of baselines) often show different color marks, which must be accurately recorded."],  
   "pattern": ["List the pattern of each sample one by one, i.e., the combination relationship between features and their different manifestations. Example: The pattern of Sample X is: Feature 1 (occurrence timing) (manifestation) (red/blue) + Feature 2 (occurrence timing) (manifestation) (red/blue) + ..."]  
}  
'''

prompt2='''
<Background>  
You are an expert in time series data analysis, capable of analyzing line plots, time-frequency images, and heatmaps. I will provide the following information:  
- Line plot, time-frequency image, and heatmap of one sample.  
- Preliminary feature summary of the label corresponding to this sample.  

Your task is to: accurately extract all feature events, their occurrence timing, manifestations, colors marked by the heatmap, and the combination relationships of different feature manifestations from the three types of plots of this sample. Then analyze whether the preliminary feature summary of the corresponding label already includes the features and patterns of the current sample. If there are any omissions, provide the missing features and patterns; if there are no omissions, return empty content in the "feature" and "pattern" fields.  

The following is key background information for these three types of plots:  
- **Line plots**: The line plot of each sample provides valuable classification features such as shape, rate of change, trend, peaks, and valleys. The numerical range of the y-axis varies by sample, and an auxiliary line y=0 is plotted to facilitate measuring the degree of deviation of data points from y=0. The amplitude and position of these features (e.g., peaks or valleys) form the basis of the analysis.  
- **Time-frequency images**: High-energy regions in time-frequency images correspond to peaks or valleys in line plots that deviate from y=0 — the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. Time-frequency images also provide rich frequency information, enabling the decomposition of signals into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as high-energy regions distributed across wider frequency components; the more intense the jitter, the more frequency components the high-energy regions may cover, thereby reflecting the dispersion of frequency distribution.  
- **Heatmaps**: Generated based on time-frequency images, heatmaps reflect the contribution of energy in different time regions and frequency components to classification. Red regions indicate that the energy of the current time period and frequency component is a typical feature of the corresponding label; blue regions indicate atypical features. The role of heatmaps is to directly associate the energy information of time-frequency images with classification labels, helping determine which feature events are crucial for classification.  


<Analysis Methods and Rules>  
Red and blue regions in the heatmap may be distributed in different time regions or different frequency components of the same time region. Therefore, the heatmap can not only reflect which time regions' events are typical/atypical features but also which frequency components of the same event are typical/atypical features. When summarizing the mapping relationship between features and heatmap colors, the following requirements must be strictly met:  
- For key features such as peaks/valleys, observe whether there is fluctuation in the plot. If there is fluctuation, you should **ignore the color mark of the bottom 0Hz frequency row in the heatmap** and focus on the color of the adjacent upper frequency component, which represents the true contribution of the feature. 0Hz represents a steady signal component and cannot reflect the contribution of fluctuating signals.  
- After identifying the mapping relationship between events and colors, describe the events by combining the line plot and time-frequency image. Examples will be provided below, and you must strictly follow the analysis ideas of the examples when analyzing similar or other events:  
  - If an oscillating peak is blue in the frequency rows above 0Hz in the heatmap (ignoring the color mark of the 0Hz row), it indicates that the amplitude of the fluctuation is an inhibitory feature; if the adjacent lower 0Hz frequency row shows a different color mark, record it in detail.  
  - If a baseline with a Y-axis numerical range of [X-Y] is marked blue in the corresponding region of the heatmap, while a baseline with a Y-axis numerical range of [A-B] is marked red in the corresponding region of the heatmap, it indicates that the Y-axis range [A-B] is the ideal numerical range for the baseline of the label, and a baseline deviating from this range will be regarded as an inhibitory feature.  

<Task>  
Please think according to the following steps:  
Step 1: Analyze the line plot, time-frequency image, and heatmap of this sample, and extract as much valuable information as possible; summarize all features appearing in the sample, their occurrence timing, manifestations (numerical range, amplitude range, shape, trend, volatility, etc.), and the heatmap colors corresponding to different manifestations of these features. If necessary, divide and describe the frequency components (e.g., the 0 Hz row, the adjacent upper frequency row, the medium-frequency region, the high-frequency region, etc.)  
Step 2: Summarize the pattern of this sample, i.e., the combination relationship between the features analyzed in Step 1 and their different manifestations. Analyze which features the sample consists of, what the manifestations of these features are, and what color each corresponds to. If necessary, divide the sequence into meaningful segments (e.g., early, middle, late stages) (Example: The pattern of the sample is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the late stage, where the early baseline is marked red in the heatmap and the late stable amplitude is marked blue in the heatmap);  
Step 3: Analyze whether the preliminary feature summary of the corresponding label already includes the features and patterns of the current sample. If there are any omissions, provide the missing features and patterns; if there are no omissions, return empty content in the "feature" and "pattern" fields.  

<Output Requirements>  
You must use JSON format to conduct a detailed analysis of the sample, covering valuable information such as trends, periodicity, baselines, peaks, valleys, stability, shape, rate of change, and the mapping relationship between features and heatmaps under different frequency components. Ensure that the output content meets all requirements and analysis rules. Only output the features and patterns that need to be supplemented; if there is no content to supplement, do not output anything for the corresponding field.  

You must return the result in JSON format, and the format must strictly follow:  
{  
   "feature": ["If the preliminary feature summary of the label does not omit the content of this sample, return empty content; otherwise, list all missing features reflected in this sample, their manifestations, and heatmap colors one by one. Features refer to baselines/peaks/valleys/others; manifestations refer to specific numerical values/stability/shape/others of these features. Different manifestations under the same label (e.g., different numerical ranges/amplitudes of baselines) often show different color marks, which must be accurately recorded."],  
   "pattern": ["If the preliminary feature summary of the label does not omit the content of this sample, return empty content; otherwise, supplement the missing pattern of the sample, i.e., the combination relationship between features and their different manifestations. Example: The pattern of Sample X is: Feature 1 (manifestation) (red/blue) + Feature 2 (manifestation) (red/blue) + ..."]  
}  
'''

prompt30='''
# <Background>
As an expert in time series data analysis, you are capable of analyzing line plots, time-frequency images, and heatmaps. I will provide you with the following information:
- The line plot, time-frequency image, and heatmap of the sample to be classified;
- Pattern summaries of typical samples that may be misclassified by black-box models.

Your task is to: accurately extract all features from the three types of plots of the sample to be classified, identify the sample's pattern, and determine whether the manifestation of each feature is a typical feature or an atypical feature (i.e., the corresponding color mark) based on the heatmap.
Based on the extracted features of the sample to be classified and the pattern summaries of typical samples that may be misclassified by black-box models, analyze whether the features of the sample to be classified are included in the feature list of typical samples prone to misclassification by black-box models.

The following is key background information for these three types of plots:
- **Line plots**: The line plot of each sample provides valuable classification features such as shape, rate of change, trend, peaks, and valleys. The numerical range of the y-axis varies by sample, and an auxiliary line y=0 is plotted to facilitate measuring the degree of deviation of data points from y=0. The amplitude and position of these features (e.g., peaks or valleys) form the basis of the analysis.
- **Time-frequency images**: High-energy regions in time-frequency images correspond to peaks or valleys in line plots that deviate from y=0 — the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. Time-frequency images also provide rich frequency information, enabling the decomposition of signals into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as high-energy regions distributed across wider frequency components; the more intense the jitter, the more frequency components the high-energy regions may cover, thereby reflecting the dispersion of frequency distribution.
- **Heatmaps**: Generated based on time-frequency images, heatmaps reflect the contribution of energy in different time regions and frequency components to classification results. Red regions indicate that the energy of the current time period and frequency component is a typical feature of the corresponding label; blue regions indicate atypical features. The role of heatmaps is to directly associate the energy information of time-frequency images with classification labels, helping determine which feature events are crucial for classification.

# <Analysis Methods and Rules>
The following rules apply **only when a heatmap is provided**; they do not apply when only a line plot and a time-frequency image (Short-Time Fourier Transform, STFT) are provided:
Red and blue regions in the heatmap may be distributed in different time regions or different frequency components of the same time region. Therefore, the heatmap can not only reflect which time regions' events are typical/atypical features but also which frequency components of the same event are typical/atypical features. When summarizing the mapping relationship between features and heatmap colors, the following requirements must be strictly met:
1.  For key features such as peaks/valleys, observe whether there is fluctuation in the line plot. If there is fluctuation, you should **ignore the color mark of the bottom 0Hz frequency row in the heatmap** and focus on the color of the adjacent upper frequency component, which represents the true contribution of the feature to classification. 0Hz represents a steady signal component and cannot reflect the contribution of fluctuating signals.
2.  After identifying the mapping relationship between events and colors, describe the events by combining the line plot and time-frequency image. Examples will be provided below, and you must strictly follow the analysis ideas of the examples when analyzing similar or other events:
    - If an oscillating peak is blue in the frequency rows above 0Hz in the heatmap (ignoring the color mark of the 0Hz row), it indicates that the amplitude of the fluctuation is an inhibitory feature; if the adjacent lower 0Hz frequency row shows a different color mark, record it in detail.
    - If a baseline with a Y-axis numerical range of [X-Y] is marked blue in the corresponding region of the heatmap, while a baseline with a Y-axis numerical range of [A-B] is marked red in the corresponding region of the heatmap, it indicates that the Y-axis range [A-B] is the ideal numerical range for the baseline of the label, and a baseline deviating from this range will be regarded as an inhibitory feature.

# <Task>
Please conduct the analysis in accordance with the following steps:
Step 1: Confirm the line plot, time-frequency image, and heatmap of the sample to be classified, and extract as much valuable information as possible; summarize all features appearing in the sample, their manifestations (numerical range, amplitude range, shape, trend, volatility, etc.), and the heatmap colors corresponding to different manifestations of these features. If necessary, divide and describe the frequency components (e.g., the 0 Hz row, the adjacent upper frequency row, the medium-frequency region, the high-frequency region, etc.).
Step 2: Summarize the pattern of this sample, i.e., the combination relationship between the features analyzed in Step 1 and their different manifestations. Analyze which features the sample consists of, what the manifestations of these features are, and what color each corresponds to. If necessary, divide the sequence into meaningful segments (e.g., early, middle, late stages) (Example: The pattern of the sample is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the late stage; where the early baseline is marked red in the heatmap and the late stable amplitude is marked blue in the heatmap).
Step 3: Analyze whether the pattern of the sample to be classified is included in the provided **list of label features and patterns**, determine whether the sample to be classified is likely to become a typical sample misclassified by the black-box model, and quantify the feature similarity between the sample to be classified and each label using an integer from 1 to 4.

The description of each level of feature similarity is as follows:
+ 1 point: No similarity — the sample to be classified has almost no association with the label; the corresponding features and heatmap color mapping relationships do not match, and there is no matching pattern or feature manifestation.
+ 2 points: Low similarity — there is weak evidence indicating a certain association between the sample to be classified and the label; there is no matching pattern with the sample to be classified, but most of the feature manifestations and heatmap color marks in the feature list match those of the sample to be classified.
+ 3 points: Moderate similarity — there is strong evidence indicating an association between the sample to be classified and the label; although there may be slight differences in pattern descriptions, a pattern that basically matches the sample to be classified can be found.
+ 4 points: High similarity — the matching degree exceeds 95%, and the label's feature and pattern list includes a pattern that completely matches the sample to be classified.

# <Output Requirements>
If you believe the sample to be classified is highly likely to become a typical sample misclassified by the black-box model, fill in the **label number corresponding to the typical misclassification pattern** in the result field; if you believe the sample to be classified has some similarity with the typical samples misclassified by the black-box model but the similarity is not significant, or if you believe the sample to be classified does not match the features of any typical misclassified sample, fill in **no** in lowercase in the result field. You must return the result in JSON format, and the format must strictly follow the specification below:
{
    "result": ["Label 0/1/… /no"],
    "score": [integer from 1 to 4],
    "rationale": ["Your complete step-by-step analysis process"]
}
'''

prompt31='''
<Background>
You are an expert in time series data analysis, with the ability to analyze **plots** and **time-frequency images**. I will provide you with the following information:
- Plots and time-frequency images of the sample to be classified;
- Feature summaries of different labels and potential patterns (combined features of different feature manifestations) that different labels may possess;
- Some reference samples with known categories.

Your task is to: accurately extract all features from the two types of charts of the sample to be classified, and identify the sample's pattern.
Based on the extracted features of the sample to be classified and the provided list of features and patterns of different labels, analyze which label's features the sample's features are more similar to, and provide reasons. Since the heatmap of the sample to be classified is not provided, you do not need to pay attention to the descriptions of the corresponding heatmap colors of features in the feature summary. Only analyze the similarity between the sample to be classified and each category based on the descriptions of features and patterns. The criterion for similarity judgment is to analyze which label's pattern list matches the pattern of the sample to be classified; if there is a match, it indicates they belong to the same label. If an accurate result cannot be obtained based on the feature summary, or there is no completely matching pattern description in the feature summary, a comparative analysis can be conducted by combining reference samples to match the most probable classification result.

The following is key background information for these two types of charts:
- **Line Plot**: The line plot of each sample provides valuable classification features such as shape, rate of change, trend, peaks, and valleys. The numerical range of the y-axis varies by sample, and an auxiliary line y=0 is plotted to facilitate measuring the degree of deviation of data points from y=0. The amplitude and position of these features (e.g., peaks or valleys) form the basis of the analysis.
- **Time-Frequency Image**: High-energy regions in time-frequency images correspond to peaks or valleys in line plots that deviate from y=0 — the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. Time-frequency images also provide rich frequency information, enabling the decomposition of signals into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as high-energy regions distributed across wider frequency components; the more intense the jitter, the more frequency components the high-energy regions may cover, thereby reflecting the dispersion of frequency distribution.

<Task>
Please think according to the following steps:
Step 1: Confirm the plot and time-frequency image of the sample to be classified, and extract as much valuable information as possible; summarize all features appearing in the sample and their manifestations (numerical range, amplitude range, shape, trend, volatility, etc.).
Step 2: Summarize the pattern of this sample, i.e., the combination relationship between the features analyzed in Step 1 and their different manifestations. Analyze which features the sample consists of and what the manifestations of these features are. If necessary, divide the sequence into meaningful segments (e.g., early, middle, late stages) (Example: The pattern of the sample is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the late stage).
Step 3: Analyze whether the "list of label features and patterns" provided to you includes the pattern of the sample to be classified, select the label with the highest matching degree as the final classification result, and quantify the feature similarity between the sample to be classified and each label using an integer from 1 to 4.

The description of each level of feature similarity is as follows:
  + 1 Point: No Similarity — the sample to be classified has almost no association with the label; there is no matching pattern or feature manifestation.
  + 2 Points: Low Similarity — there is weak evidence indicating a certain association between the sample to be classified and the label; there is no matching pattern with the sample to be classified, but most of the feature manifestations in the feature list match those of the sample to be classified.
  + 3 Points: Moderate Similarity — there is strong evidence indicating an association between the sample to be classified and the label; although there may be slight differences in pattern descriptions, a pattern that basically matches the sample to be classified can be found.
  + 4 Points: High Similarity — the matching degree exceeds 95%, and the label's feature and pattern list includes a pattern that completely matches the sample to be classified.

Step 4: Before outputting the final result, conduct further analysis by combining the reference samples, recheck the accuracy of the thinking process in each step, the fairness of the scoring logic, and whether the feature summary and reference sample information support the consistency of your classification result. After thorough deliberation, output the most accurate classification result in the end.

<Output Requirements>
You must return the result in JSON format, and the format must strictly follow:
{
   "result": ["Label 0/1/…"],
   "score": [1-4],
   "rationale": ["Your complete step-by-step analysis process"]
}
'''

prompt32 ='''
<Background>  
As an expert in time series data analysis, you are capable of analyzing line plots, time-frequency images, and heatmaps. I will provide the following information:  
- Line plot, time-frequency image, and heatmap of the sample to be classified;  
- Feature summaries of different labels and potential patterns (combined features of different feature manifestations) that different labels may possess.  
- Some reference samples with known categories.

Your task is to: accurately extract all features from the three types of plots of the sample to be classified, identify the sample's pattern, and analyze whether the manifestation of each feature is a typical feature or an atypical feature (i.e., the marked color) based on the heatmap.  
Based on the extracted features of the sample to be classified and the provided list of features and patterns of different labels, analyze which label's features the sample's features are more similar to, and provide reasons. The criterion for similarity judgment is to analyze which label's pattern list matches the pattern of the sample to be classified; if there is a match, it indicates they belong to the same label. If an accurate result cannot be obtained based on the feature summary, or there is no completely matching pattern description in the feature summary, a comparative analysis can be conducted by combining reference samples to match the most probable classification result. 

The following is key background information for these three types of plots:  
- **Line plots**: The line plot of each sample provides valuable classification features such as shape, rate of change, trend, peaks, and valleys. The numerical range of the y-axis varies by sample, and an auxiliary line y=0 is plotted to facilitate measuring the degree of deviation of data points from y=0. The amplitude and position of these features (e.g., peaks or valleys) form the basis of the analysis.  
- **Time-frequency images**: High-energy regions in time-frequency images correspond to peaks or valleys in line plots that deviate from y=0 — the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. Time-frequency images also provide rich frequency information, enabling the decomposition of signals into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as high-energy regions distributed across wider frequency components; the more intense the jitter, the more frequency components the high-energy regions may cover, thereby reflecting the dispersion of frequency distribution.  
- **Heatmaps**: Generated based on time-frequency images, heatmaps reflect the contribution of energy in different time regions and frequency components to classification. Red regions indicate that the energy of the current time period and frequency component is a typical feature of the corresponding label; blue regions indicate atypical features. The role of heatmaps is to directly associate the energy information of time-frequency images with classification labels, helping determine which feature events are crucial for classification.  


<Analysis Methods and Rules>  
The following rules apply only when a heatmap is provided; they do not apply when only a plot and time-frequency image (STFT) are provided:  
Red and blue regions in the heatmap may be distributed in different time regions or different frequency components of the same time region. Therefore, the heatmap can not only reflect which time regions' events are typical/atypical features but also which frequency components of the same event are typical/atypical features. When summarizing the mapping relationship between features and heatmap colors, the following requirements must be strictly met:  
- For key features such as peaks/valleys, observe whether there is fluctuation in the plot. If there is fluctuation, you should **ignore the color mark of the bottom 0Hz frequency row in the heatmap** and focus on the color of the adjacent upper frequency component, which represents the true contribution of the feature. 0Hz represents a steady signal component and cannot reflect the contribution of fluctuating signals.  
- After identifying the mapping relationship between events and colors, describe the events by combining the line plot and time-frequency image. Examples will be provided below, and you must strictly follow the analysis ideas of the examples when analyzing similar or other events:  
  - If an oscillating peak is blue in the frequency rows above 0Hz in the heatmap (ignoring the color mark of the 0Hz row), it indicates that the amplitude of the fluctuation is an inhibitory feature; if the adjacent lower 0Hz frequency row shows a different color mark, record it in detail.  
  - If a baseline with a Y-axis numerical range of [X-Y] is marked blue in the corresponding region of the heatmap, while a baseline with a Y-axis numerical range of [A-B] is marked red in the corresponding region of the heatmap, it indicates that the Y-axis range [A-B] is the ideal numerical range for the baseline of the label, and a baseline deviating from this range will be regarded as an inhibitory feature.  

<Task>  
Please think according to the following steps:  
Step 1: Confirm the line plot, time-frequency image, and heatmap of the sample to be classified, and extract as much valuable information as possible; summarize all features appearing in the sample, their manifestations (numerical range, amplitude range, shape, trend, volatility, etc.), and the heatmap colors corresponding to different manifestations of these features. If necessary, divide and describe the frequency components (e.g., the 0 Hz row, the adjacent upper frequency row, the medium-frequency region, the high-frequency region, etc.)  
Step 2: Summarize the pattern of this sample, i.e., the combination relationship between the features analyzed in Step 1 and their different manifestations. Analyze which features the sample consists of, what the manifestations of these features are, and what color each corresponds to. If necessary, divide the sequence into meaningful segments (e.g., early, middle, late stages) (Example: The pattern of the sample is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the late stage, where the early baseline is marked red in the heatmap and the late stable amplitude is marked blue in the heatmap);  
Step 3: Analyze whether the "list of label features and patterns" provided to you includes the pattern of the sample to be classified, select the label with the highest matching degree as the final classification result, and quantify the feature similarity between the sample to be classified and each label using an integer from 1 to 4.  

The description of each level of feature similarity is as follows:  
  + 1: No similarity — the sample to be classified has almost no association with the label; the corresponding features and heatmap color mapping relationships do not match, and there is no matching pattern or feature manifestation;  
  + 2: Low similarity — there is weak evidence indicating a certain association between the sample to be classified and the label; there is no matching pattern with the sample to be classified, but most of the feature manifestations and heatmap color marks in the feature list match those of the sample to be classified;  
  + 3: Moderate similarity — there is strong evidence indicating an association between the sample to be classified and the label; although there may be slight differences in pattern descriptions, a pattern that basically matches the sample to be classified can be found;  
  + 4: High similarity — the matching degree exceeds 95%, and the label's feature and pattern list includes a pattern that completely matches the sample to be classified.  

Step 4: Before outputting the final result, conduct further analysis by combining the reference samples, recheck the accuracy of the thinking process in each step, the fairness of the scoring logic, and whether the feature summary and reference sample information support the consistency of your classification result. After thorough deliberation, output the most accurate classification result in the end.

<Output Requirements>  
You must return the result in JSON format, and the format must strictly follow:  
{  
   "result": ["Label 0/1/…"],  
   "score": [1-4],  
   "rationale": ["Your complete step-by-step analysis process"]  
}  
'''
prompt41 = '''
<Background>
You are an expert in time series data analysis, with the ability to analyze **plots** and **time-frequency images**. I will provide you with the following information:
- Plots and time-frequency images of the sample to be classified;
- Feature summaries of different labels and the patterns (combined features of different feature manifestations) that different labels may possess;
- Classification results of the sample to be classified provided by three other assistants;
- Classification result and prediction probability value of the black-box model.

Your task is to analyze the confidence level of the model's prediction based on the classification result and logits of the black-box model, and perform the classification task by combining the classification results of the three assistants.
Specifically, analyze the features and patterns of the sample to be classified based on line plots and time-frequency images, and match them with the content in the provided feature summary. Ignore the feature descriptions such as color labels of features, and give your preliminary judgment on the sample to be classified **only based on the feature manifestations and pattern descriptions included in the feature summary**. Then, based on this preliminary judgment, analyze the classification results of the black-box model and the three assistants, and finally give the classification result you consider the most reasonable.

The following is an example of other assistants judging credibility based on the probability value of the black-box model: The logit of the model's predicted category 3 is 1.02, which is the highest among all categories, but not significantly higher than that of category 6 (logit 0.78) or category 4 (logit 0.81). This indicates that although the model has a certain level of confidence in its prediction, there is still uncertainty, and other categories may also be correct classifications.

The following is key background information for these two types of images:
- **Line Plot**: The line plot of each sample provides valuable classification features such as shape, rate of change, trend, peaks, and valleys. The numerical range of the y-axis varies by sample, and an auxiliary line y=0 is plotted to facilitate measuring the degree of deviation of data points from y=0. The amplitude and position of these features (e.g., peaks or valleys) form the basis of the analysis.
- **Time-Frequency Image**: High-energy regions in time-frequency images correspond to peaks or valleys in line plots that deviate from y=0 — the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. Time-frequency images also provide rich frequency information, enabling the decomposition of signals into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as high-energy regions distributed across wider frequency components; the more intense the jitter, the more frequency components the high-energy regions may cover, thereby reflecting the dispersion of frequency distribution.

<Task>
Please analyze according to the following steps:
Step 1: Analyze the credibility of the classification result of the black-box model based on its prediction probability. Extract the features, feature manifestations (numerical range, amplitude range, shape, trend, volatility, etc.) and pattern of the sample to be classified, and analyze which features the sample consists of and what the manifestations of these features are. If necessary, divide the sequence into meaningful segments (e.g., early, middle, late stages).
Step 2: Match the features and pattern of the sample to be classified with the content in the feature summary, and give your preliminary judgment on the sample to be classified.
Step 3: Based on your preliminary judgment of the sample to be classified, analyze the classification results of the black-box model and the three assistants, and finally give the classification result you consider the most reasonable as well as alternative results.
Step 4: Reorganize your entire thinking and reasoning process, check for any deviations from the task requirements; verify whether potential pattern features have been ignored, which results in the optimal classification result being classified as an alternative result. After a comprehensive check, give the final and most accurate classification result.

<Output Requirements>
Field Descriptions:
- result field: The final classification result of the sample to be classified;
- model field: The classification result generated by the black-box model;
- rationale field: The complete step-by-step analysis process.

You must return the result in JSON format **without adding any extra content**. The format must strictly follow the specification below:
{
   "result":["Label 0/1/…"],
   "model":["Label 0/1/…"],
   "rationale":["Your complete step-by-step analysis process"]
}
'''

prompt42  ='''
<Background>  
As an expert in time series data analysis, you are capable of analyzing line plots, time-frequency images, and heatmaps. I will provide the following information:  
- Line plot, time-frequency image, and heatmap of the sample to be classified;  
- Feature summaries of different labels and patterns (combined features of different feature manifestations) that different labels may possess;  
- Classification results of the sample to be classified from three other assistants;  
- Classification result and prediction probability value of the black-box model.  

Your task is to analyze the confidence level of the model's prediction and the credibility of the heatmap based on the classification result and logits of the black-box model, and then perform the classification task:  
- If you consider the output result of the black-box model credible, you need to accurately extract the mapping relationship between the features of the sample to be classified and the heatmap, match it with the provided feature summary, then analyze the classification results of the black-box model and the three assistants based on your preliminary judgment of the sample to be classified, and finally give the classification result you consider most reasonable.  
- If you consider the output result of the black-box model not credible, the heatmap information is likely to be incorrect. You need to ignore the information in the heatmap, analyze the features and patterns of the sample to be classified only based on the plot and time-frequency image, match them with the provided feature summary, consider only valuable information except heatmap information, give your preliminary judgment of the sample to be classified, then analyze the classification results of the black-box model and the three assistants based on your preliminary judgment of the sample to be classified, and finally give the classification result you consider most reasonable.  

The following is an example of other assistants judging credibility based on the probability value of the black-box model: The logit of the model's predicted category 3 is 1.02, which is the highest among all categories, but not significantly higher than category 6 (logit 0.78) or category 4 (logit 0.81). This indicates that although the model has certain confidence in its prediction, there is still uncertainty, and other categories may also be valid.  

The following is key background information for these three types of plots:  
- **Line plots**: The line plot of each sample provides valuable classification features such as shape, rate of change, trend, peaks, and valleys. The numerical range of the y-axis varies by sample, and an auxiliary line y=0 is plotted to facilitate measuring the degree of deviation of data points from y=0. The amplitude and position of these features (e.g., peaks or valleys) form the basis of the analysis.  
- **Time-frequency images**: High-energy regions in time-frequency images correspond to peaks or valleys in line plots that deviate from y=0 — the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. Time-frequency images also provide rich frequency information, enabling the decomposition of signals into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as high-energy regions distributed across wider frequency components; the more intense the jitter, the more frequency components the high-energy regions may cover, thereby reflecting the dispersion of frequency distribution.  
- **Heatmaps**: Generated based on time-frequency images, heatmaps reflect the contribution of energy in different time regions and frequency components to classification. Red regions indicate that the energy of the current time period and frequency component is a typical feature of the corresponding label; blue regions indicate atypical features. The role of heatmaps is to directly associate the energy information of time-frequency images with classification labels, helping determine which feature events are crucial for classification.  


<Analysis Methods and Rules>  
The following rules apply only when the heatmap is credible:  
Red and blue regions in the heatmap may be distributed in different time regions or different frequency components of the same time region. Therefore, the heatmap can not only reflect which time regions' events are typical/atypical features but also which frequency components of the same event are typical/atypical features. When summarizing the mapping relationship between features and heatmap colors, the following requirements must be strictly met:  
- For key features such as peaks/valleys, observe whether there is fluctuation in the plot. If there is fluctuation, you should **ignore the color mark of the bottom 0Hz frequency row in the heatmap** and focus on the color of the adjacent upper frequency component, which represents the true contribution of the feature. 0Hz represents a steady signal component and cannot reflect the contribution of fluctuating signals.  
- After identifying the mapping relationship between events and colors, describe the events by combining the line plot and time-frequency image. Examples will be provided below, and you must strictly follow the analysis ideas of the examples when analyzing similar or other events:  
  - If an oscillating peak is blue in the frequency rows above 0Hz in the heatmap (ignoring the color mark of the 0Hz row), it indicates that the amplitude of the fluctuation is an inhibitory feature; if the adjacent lower 0Hz frequency row shows a different color mark, record it in detail.  
  - If a baseline with a Y-axis numerical range of [X-Y] is marked blue in the corresponding region of the heatmap, while a baseline with a Y-axis numerical range of [A-B] is marked red in the corresponding region of the heatmap, it indicates that the Y-axis range [A-B] is the ideal numerical range for the baseline of the label, and a baseline deviating from this range will be regarded as an inhibitory feature.  

<Task>  
Please think according to the following steps:  
Step 1: Analyze the credibility of the classification result of the black-box model and the heatmap based on the prediction probability of the black-box model.  
  - If not credible, ignore all provided information about the heatmap, extract the features, manifestations (numerical range, amplitude range, shape, trend, volatility, etc.) and pattern of the sample to be classified, and analyze which features the sample to be classified consists of and what the manifestations of these features are. If necessary, divide the sequence into meaningful segments (e.g., early, middle, late stages);  
  - If credible, extract the features, manifestations (numerical range, amplitude range, shape, trend, volatility, etc.) of the sample to be classified and the heatmap colors corresponding to different manifestations of these features in combination with heatmap information. If necessary, divide and describe the frequency components (e.g., the 0 Hz row, the adjacent upper frequency row, the medium-frequency region, the high-frequency region, etc.).Then analyze the pattern of the sample to be classified, including which features the sample consists of, what the manifestations of these features are, and what color each corresponds to. If necessary, divide the sequence into meaningful segments (e.g., early, middle, late stages).  

Step 2: Match the features and pattern of the sample to be classified with the content in the feature summary, and give your preliminary understanding of the sample to be classified.  

Step 3: Analyze the classification results of the black-box model and the three assistants based on your preliminary judgment of the sample to be classified, and finally give the classification result you consider most reasonable and alternative results;  

Step 4: Reanalyze your entire thinking and reasoning process, check for any deviations from the task requirements; check if potential pattern features have been ignored, leading to the optimal classification result being placed in the alternative results, and give the final and most accurate classification result after a complete check.  

<Output Requirements>  
Field Description:  
- result field: The final classification result of the sample to be classified;  
- model field: The classification result generated by the black-box model;  
- rationale field: The complete step-by-step analysis process.  

You must return the result in JSON format, and no additional content may be added. The format must strictly follow:  
{  
   "result":["Label 0/1/…"],  
   "model":["Label 0/1/…"],  
   "rationale":["Your complete step-by-step analysis process"]  
}
'''