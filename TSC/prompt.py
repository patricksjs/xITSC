prompt11 = '''
<Background>
You are an expert in time series data analysis, capable of analyzing line plots, time-frequency images, and heatmaps. I will provide line plots, time-frequency images, and heatmaps of multiple samples under the same label.

Your task is to accurately extract all feature events from these three types of charts and analyze whether each event is a typical or atypical feature based on the heatmap. A feature event refers to the specific manifestations in the line plot and time-frequency image of the colored (red or blue) time-frequency regions in the heatmap:
- In the line plot, feature events may manifest as peaks, valleys, trend changes, jitters, morphological alterations, or other characteristics.
- In the time-frequency image, feature events may manifest as high-energy regions in high/low frequency bands (e.g., high-energy regions occupying adjacent or discrete frequency components within a certain time window).
- In the heatmap, events are classified as typical features (red) or atypical features (blue) based on color coding.

The following is core background information about these three types of charts:
- **Line Plot**: The line plot of each sample provides classification-relevant features such as morphology, rate of change, trend, peaks, and valleys. Since the Y-axis value ranges of different samples vary, an auxiliary line y=0 is plotted in the chart to facilitate measuring the deviation of data points relative to y=0 and assist in analyzing the magnitude relationship of peaks/valleys.
- **Time-Frequency Image**: High-energy regions in the time-frequency image correspond to peaks or valleys of the line plot that deviate from y=0—the greater the deviation from y=0, the higher the energy in the corresponding time window of the time-frequency image. The time-frequency image also provides rich frequency information, allowing signal decomposition into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as a high-energy region distributed across a wider range of frequency components; the more severe the jitter, the more frequency components the high-energy region may cover, thereby reflecting the degree of frequency distribution dispersion.
- **Heatmap**: Generated based on the time-frequency image, it reflects the contribution of energy in different time windows and frequency components to the classification of the black-box model. Red regions indicate that the energy of the current time window and frequency component is a typical feature of the corresponding label; blue regions indicate that it is an atypical feature. The heatmap correlates the energy information of the time-frequency image directly with classification labels, helping to determine which feature events are critical for classification.

<Analysis Methods and Rules>
The heatmap reflects the contribution of features in different regions to the classification of the black-box model. Therefore, features in different regions may exhibit the following scenarios:
1. If red markings dominate a region (with a small amount of blue markings allowed), the black-box model considers the features in this region to be typical features of the current label;
2. If blue markings dominate a region (with a small amount of red markings allowed), the black-box model considers the features in this region to be atypical features of the current label, exhibiting characteristics more similar to those of other labels;
3. If a region has a mixture of red and blue markings, the black-box model is confused about the features in this region, which exhibit indistinct similarities to those of other labels.

Red and blue regions in the heatmap may be distributed in different time windows or across different frequency components within the same time window. Therefore, the heatmap not only reflects which time window events are typical/atypical features but also indicates which frequency components of the same event are typical/atypical features. When describing features, it is necessary to strictly determine whether the manifestation of a feature is a typical feature, an atypical feature, or an indistinct ambiguous feature.

<Task Requirements>
Please think step by step:
Step 1: Analyze the line plot, time-frequency image, and heatmap of each sample under the target label to extract as much valuable information as possible; summarize all features present in each sample, their manifestations (value range, amplitude range, morphology, trend, volatility, etc.), the heatmap color types corresponding to different manifestations of these features, and determine whether these feature manifestations are typical features, atypical features, or indistinct ambiguous features. If necessary, frequency components can be divided and described (e.g., 0Hz row, adjacent upper frequency rows, intermediate frequency region, high frequency region, etc.).
Step 2: Summarize the feature pattern of each sample, i.e., the combination relationship between features analyzed in Step 1 and their different manifestations. Analyze which features constitute each sample, what their manifestations are, and the corresponding color and feature type of each. If necessary, the sequence can be divided into meaningful phases (e.g., early stage, middle stage, late stage). (Example: The feature pattern of Sample X is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the late stage; among them, the early-stage baseline is marked red in the heatmap, and the stable late-stage amplitude is marked blue).
Step 3: Inductively analyze the results of each reference sample to summarize the complete feature list and pattern features of the label.

<Output Requirements>
You must present a detailed analysis of the label to be summarized in JSON format. The summary content shall be no less than 400 words, covering valuable information such as trends, periodicity, baselines, peaks, valleys, stability, morphology, and rate of change, as well as the mapping relationship between features and heatmaps across different frequency components. Ensure that the output content meets all requirements and analysis rules.

The output results must be in JSON format and strictly comply with the following specifications:
{
    "feature": ["List all features reflected by the samples one by one, including the occurrence time of features, their manifestations, and corresponding heatmap colors. Features are classified into baseline/peak/valley/other types; manifestations refer to specific values/stability/morphology/other attributes of these features. Different manifestations of the same feature under the same label (e.g., different value ranges/amplitude of baselines) often correspond to different color markings and must be accurately recorded"],
    "pattern": ["List the feature pattern of each sample one by one, i.e., the combination relationship between features and their different manifestations. Example: The feature pattern of Sample X is: Feature 1 (occurrence time) (manifestation) (red/blue) (heatmap color interpretation) + Feature 2 (occurrence time) (manifestation) (red/blue) (heatmap color interpretation) + ……"]
}
'''

prompt12 = '''
# Prompt 12
<Background>
You are an expert in time series data analysis, capable of analyzing line plots, time-frequency images, and heatmaps. I will provide the following information:
- Line plots, time-frequency images, and heatmaps of multiple samples under the same label; these samples are classification boundary samples corresponding to the label and can cause misclassification by black-box models.
- Classification results and log probabilities of these samples generated by the black-box model.

Your task is to analyze which feature events in these boundary samples lead to model misclassification and what the manifestation forms of the patterns of these boundary samples are.
- In line plots, feature events may manifest as peaks, valleys, trend changes, jitters, morphological alterations, or other characteristics.
- In time-frequency images, feature events may manifest as high-energy regions in high-frequency/low-frequency bands (e.g., high-energy regions occupying adjacent or discrete frequency components within a certain time region).
- In heatmaps, events are determined to be typical features (red) or atypical features (blue) based on color.

The following is the core background information of these three types of charts:
- **Line Plot**: The line plot of each sample can provide classification-relevant features such as morphology, rate of change, trend, peaks, and valleys. Since the Y-axis value ranges of different samples vary, an auxiliary line y=0 is drawn in the plot to facilitate measuring the deviation degree of data points relative to y=0, helping to analyze the magnitude relationship of peaks/valleys.
- **Time-Frequency Image**: High-energy regions in the time-frequency image correspond to peaks or valleys that deviate from y=0 in the line plot—the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. The time-frequency image also provides rich frequency information and can decompose signals into different frequency components. For example, if a peak/valley in the line plot has jitters, the time-frequency image will accurately capture it as a high-energy region distributed over a wider range of frequency components; the more severe the jitters, the more frequency components the high-energy region may cover, thereby reflecting the dispersion degree of frequency distribution.
- **Heatmap**: Generated based on the time-frequency image, it reflects the contribution degree of energy in different time regions and frequency components to the classification of the black-box model. Red regions indicate that the energy of the current time period and frequency component is a typical feature of the corresponding label; blue regions indicate that it is an atypical feature. The role of the heatmap is to directly associate the energy information of the time-frequency image with classification labels, helping to determine which feature events are crucial for classification.

<Analysis Methods and Rules>
The heatmap reflects the contribution degree of features in different regions to the classification of the black-box model. Therefore, features in different regions may present the following situations:
1. If red markings dominate the features in a region (allowing a small amount of blue markings), it means the black-box model considers the features in this region as typical features of the current label.
2. If blue markings dominate the features in a region (allowing a small amount of red markings), it means the black-box model considers the features in this region as atypical features of the current label, showing characteristics more similar to other labels.
3. If the features in a region are mixed with red and blue, it means the black-box model is confused about the features in this region, showing similar features that are difficult to distinguish from other labels.

Red and blue regions in the heatmap may be distributed in different time regions or different frequency components of the same time region. Therefore, the heatmap can not only reflect which time-region events are typical/atypical features but also indicate which frequency components of the same event are typical/atypical features. When describing features, it is necessary to strictly judge whether the manifestation form of a feature is a typical feature, an atypical feature, or an indistinguishable ambiguous feature.

<Task Requirements>
Please conduct the analysis in accordance with the following steps:
Step 1: Analyze the line plot, time-frequency image, and heatmap of each sample under the target label to extract as much valuable information as possible; summarize all features appearing in each sample and their manifestation forms (value range, amplitude range, morphology, trend, volatility, etc.), the heatmap color types corresponding to these different manifestation forms of features, and judge whether these feature manifestation forms belong to typical features, atypical features, or indistinguishable ambiguous features. If necessary, frequency components can be divided and described (e.g., 0Hz row, adjacent upper frequency rows, intermediate frequency region, high frequency region, etc.), and the features that you think cause misclassification by the black-box model should be identified and recorded.

Step 2: Summarize the feature pattern of each sample, i.e., the combination relationship between the features analyzed in Step 1 and their different manifestation forms. Analyze which features each sample consists of, what the manifestation forms of these features are, and what color and feature type each corresponds to. If necessary, the sequence can be divided into meaningful phases (e.g., early phase, middle phase, late phase). (Example: The feature pattern of Sample X is: excessive baseline deviation in the early phase (below -1) + stable amplitude in the late phase; among them, the early baseline is marked red in the heatmap, and the late stable amplitude is marked blue.)

Step 3: Combine the predicted log probabilities of each sample by the black-box model to reflect on your analysis results, analyze why these samples lead to model misclassification, evaluate which features and patterns are classification boundary features that are prone to causing misclassification by the black-box model based on heatmap information and log probabilities. Finally, summarize the features that are prone to causing misclassification by the black-box model and the pattern of each boundary sample; there is no need to summarize the features regarded as typical by the model.

<Output Requirements>
You must conduct a detailed analysis of the label to be summarized in JSON format. The summary content shall not be less than 400 words, covering valuable information such as trends, periodicity, baselines, peaks, valleys, stability, morphology, and rate of change, as well as the mapping relationship between features and heatmaps under different frequency components. Ensure that the output content meets all requirements and analysis rules.

The output result must be in JSON format and strictly follow the following specifications:
{
    "feature": ["List all features that cause misclassification reflected by the samples one by one, including the occurrence time, manifestation form and corresponding heatmap color of the features. Features are divided into baseline/peak/valley/other types; manifestation forms refer to the specific values/stability/morphology/other attributes of these features. Different manifestation forms of the same feature under the same label (e.g., different value ranges/amplitude of the baseline) often correspond to different color markings, which need to be accurately recorded"],
    "pattern": ["List the feature pattern of each sample one by one, i.e., the combination relationship between features and their different manifestation forms. Example: The feature pattern of Sample X is: Feature 1 (occurrence time) (manifestation form) (red/blue) (heatmap color interpretation) + Feature 2 (occurrence time) (manifestation form) (red/blue) (heatmap color interpretation) + ……"]
}
'''

prompt21='''
<Background>
You are an expert in time series data analysis, capable of analyzing line plots, time-frequency images, and heatmaps. I will provide the following information:
- Line plot, time-frequency image, and heatmap of a single sample;
- Preliminary feature summary of the label corresponding to this sample.

Your task is to accurately extract all feature events, their occurrence times, manifestations, heatmap marking colors, and the combination relationship between different feature manifestations from the three types of charts of the sample. Then analyze whether the preliminary feature summary of the corresponding label covers the features and patterns of the current sample; if there are omissions, supplement the missing features and patterns; if there are no omissions, return empty content in the "feature" and "pattern" fields.

The following is core background information about these three types of charts:
- **Line Plot**: The line plot of each sample provides classification-relevant features such as morphology, rate of change, trend, peaks, and valleys. Since the Y-axis value ranges of different samples vary, an auxiliary line y=0 is plotted in the chart to facilitate measuring the deviation of data points relative to y=0 and assist in analyzing the magnitude relationship of peaks/valleys.
- **Time-Frequency Image**: High-energy regions in the time-frequency image correspond to peaks or valleys of the line plot that deviate from y=0—the greater the deviation from y=0, the higher the energy in the corresponding time window of the time-frequency image. The time-frequency image also provides rich frequency information, allowing signal decomposition into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as a high-energy region distributed across a wider range of frequency components; the more severe the jitter, the more frequency components the high-energy region may cover, thereby reflecting the degree of frequency distribution dispersion.
- **Heatmap**: Generated based on the time-frequency image, it reflects the contribution of energy in different time windows and frequency components to the classification of the black-box model. Red regions indicate that the energy of the current time window and frequency component is a typical feature of the corresponding label; blue regions indicate that it is an atypical feature. The heatmap correlates the energy information of the time-frequency image directly with classification labels, helping to determine which feature events are critical for classification.

<Analysis Methods and Rules>
The heatmap reflects the contribution of features in different regions to the classification of the black-box model. Therefore, features in different regions may exhibit the following scenarios:
1. If red markings dominate a region (with a small amount of blue markings allowed), the black-box model considers the features in this region to be typical features of the current label;
2. If blue markings dominate a region (with a small amount of red markings allowed), the black-box model considers the features in this region to be atypical features of the current label, exhibiting characteristics more similar to those of other labels;
3. If a region has a mixture of red and blue markings, the black-box model is confused about the features in this region, which exhibit indistinct similarities to those of other labels.

Red and blue regions in the heatmap may be distributed in different time windows or across different frequency components within the same time window. Therefore, the heatmap not only reflects which time window events are typical/atypical features but also indicates which frequency components of the same event are typical/atypical features. When describing features, it is necessary to strictly determine whether the manifestation of a feature is a typical feature, an atypical feature, or an indistinct ambiguous feature.

<Task Requirements>
Please think step by step:
Step 1: Analyze the line plot, time-frequency image, and heatmap of the sample to extract as much valuable information as possible; summarize all features present in the sample, their manifestations (value range, amplitude range, morphology, trend, volatility, etc.), the heatmap color types corresponding to different manifestations of these features, and determine whether these feature manifestations are typical features, atypical features, or indistinct ambiguous features. If necessary, frequency components can be divided and described (e.g., 0Hz row, adjacent upper frequency rows, intermediate frequency region, high frequency region, etc.).
Step 2: Summarize the feature pattern of the sample, i.e., the combination relationship between features analyzed in Step 1 and their different manifestations. Analyze which features constitute the sample, what their manifestations are, and the corresponding color and feature type of each. If necessary, the sequence can be divided into meaningful phases (e.g., early stage, middle stage, late stage). (Example: The feature pattern of Sample X is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the late stage; among them, the early-stage baseline is marked red in the heatmap, and the stable late-stage amplitude is marked blue).
Step 3: Analyze whether the preliminary feature summary of the corresponding label covers the features and patterns of the current sample; if there are omissions, supplement the missing features and patterns; if there are no omissions, return empty content in the "feature" and "pattern" fields.

<Output Requirements>
You must present a detailed analysis of the sample in JSON format, covering valuable information such as trends, periodicity, baselines, peaks, valleys, stability, morphology, and rate of change, as well as the mapping relationship between features and heatmaps across different frequency components. Ensure that the output content meets all requirements and analysis rules. Only output the missing features and patterns that need to be supplemented; if there is no content to supplement, leave the corresponding fields empty.

The output results must be in JSON format and strictly comply with the following specifications:
{
    "feature": ["If the preliminary feature summary of the label does not omit the relevant content of the sample, return empty; otherwise, list all missing features reflected by the sample one by one, their manifestations, and heatmap color types. Features are classified into baseline/peak/valley/other types; manifestations refer to specific values/stability/morphology/other attributes of these features. Different manifestations of the same feature under the same label (e.g., different value ranges/amplitude of baselines) often correspond to different color markings and must be accurately recorded"],
    "pattern": ["If the preliminary feature summary of the label does not omit the relevant content of the sample, return empty; otherwise, supplement the missing feature pattern of the sample, i.e., the combination relationship between features and their different manifestations. Example: The feature pattern of Sample X is: Feature 1 (manifestation) (red/blue) (heatmap color interpretation) + Feature 2 (manifestation) (red/blue) (heatmap color interpretation) + ……"]
}
'''

prompt22 ='''
<Background>
You are an expert in time series data analysis, capable of analyzing line plots, time-frequency images, and heatmaps. I will provide the following information:
- Line plot, time-frequency image, and heatmap of a single sample; this sample is a classification boundary sample of the corresponding label and will cause misclassification by the black-box model.
- The classification result and logarithmic probability of this sample generated by the black-box model.
- Preliminary feature summary of the label corresponding to this sample.

Your task is to accurately extract all feature events from these three types of charts, analyze whether each event is a typical or atypical feature based on the heatmap, and further identify which features and patterns make a sample a classification boundary sample. Finally, analyze whether the preliminary feature summary of the corresponding label covers the features and patterns of the current sample; if there are omissions, supplement the missing features and patterns; if there are no omissions, return empty content in the "feature" and "pattern" fields. A feature event refers to the specific manifestations in the line plot and time-frequency image of the colored (red or blue) time-frequency regions in the heatmap:
- In the line plot, feature events may manifest as peaks, valleys, trend changes, jitters, morphological alterations, or other characteristics.
- In the time-frequency image, feature events may manifest as high-energy regions in high/low frequency bands (e.g., high-energy regions occupying adjacent or discrete frequency components within a certain time window).
- In the heatmap, events are classified as typical features (red) or atypical features (blue) based on color coding.

The following is core background information about these three types of charts:
- **Line Plot**: The line plot of each sample provides classification-relevant features such as morphology, rate of change, trend, peaks, and valleys. Since the Y-axis value ranges of different samples vary, an auxiliary line y=0 is plotted in the chart to facilitate measuring the deviation of data points relative to y=0 and assist in analyzing the magnitude relationship of peaks/valleys.
- **Time-Frequency Image**: High-energy regions in the time-frequency image correspond to peaks or valleys of the line plot that deviate from y=0—the greater the deviation from y=0, the higher the energy in the corresponding time window of the time-frequency image. The time-frequency image also provides rich frequency information, allowing signal decomposition into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as a high-energy region distributed across a wider range of frequency components; the more severe the jitter, the more frequency components the high-energy region may cover, thereby reflecting the degree of frequency distribution dispersion.
- **Heatmap**: Generated based on the time-frequency image, it reflects the contribution of energy in different time windows and frequency components to the classification of the black-box model. Red regions indicate that the energy of the current time window and frequency component is a typical feature of the corresponding label; blue regions indicate that it is an atypical feature. The heatmap correlates the energy information of the time-frequency image directly with classification labels, helping to determine which feature events are critical for classification.

<Analysis Methods and Rules>
The heatmap reflects the contribution of features in different regions to the classification of the black-box model. Therefore, features in different regions may exhibit the following scenarios:
1. If red markings dominate a region (with a small amount of blue markings allowed), the black-box model considers the features in this region to be typical features of the current label;
2. If blue markings dominate a region (with a small amount of red markings allowed), the black-box model considers the features in this region to be atypical features of the current label, exhibiting characteristics more similar to those of other labels;
3. If a region has a mixture of red and blue markings, the black-box model is confused about the features in this region, which exhibit indistinct similarities to those of other labels.

Red and blue regions in the heatmap may be distributed in different time windows or across different frequency components within the same time window. Therefore, the heatmap not only reflects which time window events are typical/atypical features but also indicates which frequency components of the same event are typical/atypical features. When describing features, it is necessary to strictly determine whether the manifestation of a feature is a typical feature, an atypical feature, or an indistinct ambiguous feature.

<Task Requirements>
Please think step by step:
Step 1: Analyze the line plot, time-frequency image, and heatmap of the sample to extract as much valuable information as possible; summarize all features present in the sample, their manifestations (value range, amplitude range, morphology, trend, volatility, etc.), the heatmap color types corresponding to different manifestations of these features, and determine whether these feature manifestations are typical features, atypical features, or indistinct ambiguous features. If necessary, frequency components can be divided and described (e.g., 0Hz row, adjacent upper frequency rows, intermediate frequency region, high frequency region, etc.).
Step 2: Summarize the feature pattern of the sample, i.e., the combination relationship between features analyzed in Step 1 and their different manifestations. Analyze which features constitute the sample, what their manifestations are, and the corresponding color and feature type of each. If necessary, the sequence can be divided into meaningful phases (e.g., early stage, middle stage, late stage). (Example: The feature pattern of Sample X is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the late stage; among them, the early-stage baseline is marked red in the heatmap, and the stable late-stage amplitude is marked blue).
Step 3: Reflect on your analysis results by combining the predicted logarithmic probability of the sample generated by the black-box model, analyze why this sample causes misclassification by the model, evaluate which features and patterns are classification boundary features prone to causing misclassification by the black-box model based on heatmap information and logarithmic probability.
Step 4: Analyze whether the preliminary feature summary of the corresponding label covers the features and patterns of the current sample; if there are omissions, supplement the missing features and patterns; if there are no omissions, return empty content in the "feature" and "pattern" fields.

<Output Requirements>
You must present a detailed analysis of the label to be summarized in JSON format. The summary content shall be no less than 400 words, covering valuable information such as trends, periodicity, baselines, peaks, valleys, stability, morphology, and rate of change, as well as the mapping relationship between features and heatmaps across different frequency components. Ensure that the output content meets all requirements and analysis rules.

The output results must be in JSON format and strictly comply with the following specifications:
{
    "feature": ["If the preliminary feature summary of the label does not omit the relevant content of the sample, return empty; otherwise, list all features that cause misclassification reflected by the sample one by one, including the occurrence time of features, their manifestations, and corresponding heatmap colors. Features are classified into baseline/peak/valley/other types; manifestations refer to specific values/stability/morphology/other attributes of these features. Different manifestations of the same feature under the same label (e.g., different value ranges/amplitude of baselines) often correspond to different color markings and must be accurately recorded"],
    "pattern": ["If the preliminary feature summary of the label does not omit the relevant content of the sample, return empty; otherwise, list the feature pattern of the sample one by one, i.e., the combination relationship between features and their different manifestations. Example: The feature pattern of Sample X is: Feature 1 (occurrence time) (manifestation) (red/blue) (heatmap color interpretation)  (logits of black-box model) + Feature 2 (occurrence time) (manifestation) (red/blue) (heatmap color interpretation) +  (logits of black-box model) ……"]
}
'''

prompt30='''
# <Background>
As an expert in time series data analysis, you are capable of analyzing line plots, time-frequency images, and heatmaps. I will provide you with the following information:
- Line plot, time-frequency image, and heatmap of the sample to be classified;
- Pattern summary of typical samples that are prone to misclassification by the black-box model.

Your task is to accurately extract all features from the three types of charts of the sample to be classified, clarify the feature pattern of the sample, and determine whether the manifestation of each feature is a typical feature or an atypical feature based on the heatmap (i.e., the corresponding color marking).
Combined with the extracted features of the sample to be classified and the pattern summary of typical samples prone to misclassification by the black-box model, analyze whether the features of the sample to be classified are included in the feature list of typical samples that are easily misclassified by the black-box model.

The following is the core background information about these three types of charts:
- **Line Plot**: The line plot of each sample can provide classification-relevant features such as morphology, rate of change, trend, peaks, and valleys. Since the Y-axis value ranges of different samples vary, an auxiliary line y=0 is plotted in the chart to facilitate measuring the deviation of data points relative to y=0, which helps analyze the magnitude relationship of peaks and valleys.
- **Time-Frequency Image**: High-energy regions in the time-frequency image correspond to the peaks or valleys of the line plot that deviate from y=0—the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. The time-frequency image also provides rich frequency information and can decompose the signal into different frequency components: for example, if a peak or valley in the line plot has jitters, the time-frequency image will accurately capture it as a high-energy region distributed over a wider range of frequency components; the more severe the jitters, the more frequency components the high-energy region may cover, thereby reflecting the degree of dispersion of the frequency distribution.
- **Heatmap**: Generated based on the time-frequency image, it reflects the contribution of energy in different time regions and frequency components to the classification of the black-box model. Red regions indicate that the energy of the current time region and frequency component is a typical feature of the corresponding label; blue regions indicate that it is an atypical feature. The function of the heatmap is to directly correlate the energy information of the time-frequency image with classification labels, helping to determine which feature events are crucial for classification.

<Analysis Methods and Rules>
The heatmap reflects the contribution of features in different regions to the classification of the black-box model. Therefore, features in different regions may present the following scenarios:
1. If red markings dominate a region (with a small amount of blue markings allowed), it means the black-box model regards the features of this region as typical features of the current label;
2. If blue markings dominate a region (with a small amount of red markings allowed), it means the black-box model regards the features of this region as atypical features of the current label, which show more similarities to the features of other labels;
3. If a region has a mixture of red and blue markings, it means the black-box model is confused about the features of this region, which show indistinct similarities to those of other labels.

The red and blue regions in the heatmap may be distributed in different time regions or across different frequency components within the same time region. Therefore, the heatmap can not only reflect which time-region events are typical/atypical features, but also indicate which frequency components of the same event are typical/atypical features. When describing features, it is necessary to strictly judge whether the manifestation of a feature is a typical feature, an atypical feature, or an indistinct ambiguous feature.

# <Task Requirements>
Please think step by step:
Step 1: Analyze the line plot, time-frequency image, and heatmap of the sample to extract as much valuable information as possible; summarize all features appearing in the sample and their manifestations (value range, amplitude range, morphology, trend, volatility, etc.), the heatmap color types corresponding to different manifestations of these features, and judge whether these feature manifestations are typical features, atypical features, or indistinct ambiguous features. If necessary, frequency components can be divided and described (e.g., 0Hz row, adjacent upper frequency rows, intermediate frequency region, high frequency region, etc.).
Step 2: Summarize the feature pattern of the sample, i.e., the combination relationship between the features analyzed in Step 1 and their different manifestations. Analyze which features constitute each sample, what the manifestations of these features are, and what color and feature type each corresponds to. If necessary, the sequence can be divided into meaningful phases (e.g., early stage, middle stage, late stage). (Example: The feature pattern of Sample X is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the late stage; among them, the early-stage baseline is marked red in the heatmap, and the stable late-stage amplitude is marked blue)
Step 3: Analyze whether the feature pattern of the sample to be classified is included in the provided **label feature and pattern list**, and determine whether the sample to be classified is likely to become a typical sample prone to misclassification by the black-box model.

# <Output Requirements>
If you believe that the sample to be classified is highly likely to become a typical misclassification sample of the black-box model, fill in the **label number corresponding to the typical misclassification pattern** in the result field; if you believe that the sample to be classified has a certain similarity to the typical misclassification samples of the black-box model but the similarity is not significant, or the features of the sample to be classified do not match any typical misclassification samples, fill in lowercase **no** in the result field. The output results must be in JSON format and strictly comply with the following specifications:
{
    "result": ["label 0/1/……/no"],
    "score": [integer from 1 to 4],
    "rationale": ["Your analysis process of the sample, focusing on exploring whether the sample to be classified is a boundary sample that causes misclassification by the black-box model, as well as the approximate logits values that the black-box model typically outputs for such boundary samples."]
}
'''

prompt31='''
<Background>
You are an expert in time series data analysis, capable of analyzing **line plots** and **time-frequency images**. I will provide you with the following information:
- Line plot and time-frequency image of the sample to be classified;
- Feature summaries of different labels, as well as potential feature patterns that different labels may have (combined features of different feature manifestations); among them, features marked red in the heatmap are typical features of the corresponding label, features marked blue are atypical features of the corresponding label, and features with a mixture of red and blue are similar features that are indistinguishable from other labels;
- Some reference samples with known categories;

Your task is to accurately extract all features from the line plot and time-frequency image of the sample to be classified, and clarify the feature pattern of the sample.
Combined with the extracted features of the sample to be classified, some reference samples with known categories, and the provided feature and pattern lists of different labels, analyze which label's features the sample's features are more similar to and explain the reasons. Since the sample to be classified is identified as a boundary sample that may cause misclassification by the black-box model, it may exhibit features similar to those of other labels. When comparing with reference sample images, care should be taken not to regard misclassification features as direct classification basis.

The following is core background information about these two types of charts:
- **Line Plot**: The line plot of each sample provides classification-relevant features such as morphology, rate of change, trend, peaks, and valleys. The Y-axis value ranges of different samples vary, and an auxiliary line y=0 is plotted in the chart to facilitate measuring the deviation of data points relative to y=0. The amplitude and position of these features (e.g., peaks or valleys) are the core basis for analysis.
- **Time-Frequency Image**: High-energy regions in the time-frequency image correspond to peaks or valleys of the line plot that deviate from y=0—the greater the deviation from y=0, the higher the energy in the corresponding time window of the time-frequency image. The time-frequency image also provides rich frequency information, allowing signal decomposition into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as a high-energy region distributed across a wider range of frequency components; the more severe the jitter, the more frequency components the high-energy region may cover, thereby reflecting the degree of frequency distribution dispersion.

<Task Requirements>
Please think step by step:
Step 1: Confirm the line plot and time-frequency image of the sample to be classified, extract as much valuable information as possible; summarize all features present in the sample and their manifestations (value range, amplitude range, morphology, trend, volatility, etc.).
Step 2: Summarize the feature pattern of the sample, i.e., the combination relationship between features analyzed in Step 1 and their different manifestations. Analyze which features constitute the sample and what their manifestations are. If necessary, the sequence can be divided into meaningful phases (e.g., early stage, middle stage, late stage). (Example: The feature pattern of this sample is: excessive baseline deviation in the early stage (below -1) + stable amplitude in the late stage).
Step 3: Analyze whether the provided **label feature and pattern list** includes the feature pattern of the sample to be classified, select the label with the highest matching degree as the final classification result, and quantify the feature similarity between the sample to be classified and each label with an integer from 1 to 4.
Step 4: Before outputting the final result, conduct further analysis in combination with reference samples to recheck the accuracy of the thinking process in each step, the fairness of the scoring logic, and the consistency between the feature summary, reference sample information, and the classification result. After comprehensive and careful verification, output the most accurate classification result, and quantify the feature similarity between the sample to be classified and each label using an integer score from 1 to 4.

The description of each similarity level is as follows:
  + 1 point: **No similarity** — The sample to be classified has almost no correlation with the feature summary of the label, with no matching significant features or feature patterns; it shows significant differences in visual patterns from the reference samples of the corresponding label, and key feature points are inconsistent.
  + 2 points: **Low similarity** — There are 1-2 weakly correlated feature matches, but no complete feature pattern match; the conformity of feature manifestations is less than 50%. It has some similarity in basic shape to the reference samples, but there are obvious differences in key feature regions.
  + 3 points: **Moderate similarity** — There are multiple feature matches (≥3), with a basically matching feature pattern; the conformity of feature manifestations is between 50% and 80%. It has high similarity in key feature regions to the reference samples of the corresponding label, and the shape pattern is basically consistent.
  + 4 points: **High similarity** — The main features (≥80%) in the feature list are highly matched, with a completely matching feature pattern; the conformity of feature manifestations is >80%. It is highly similar in visual terms to the reference samples of the corresponding label, and key feature regions are almost identical.

<Output Requirements>
The output results must be in JSON format and strictly comply with the following specifications:
{
    "result": ["label 0/1/……"],
    "score": [integer from 1 to 4],
    "rationale": ["Your complete step-by-step analysis process"]
}
'''

prompt32 ='''
<Background>
As a time series data analysis expert, you are capable of analyzing **line plots**, **time-frequency images**, and heatmaps. I will provide you with the following information:
- Line plot, time-frequency image, and heatmap of the sample to be classified;
- Feature summaries of different labels, as well as potential feature patterns (combinations of different feature manifestations) that different labels may possess;
- Partial reference samples with known labels.

Your task is to: accurately extract all features from the three types of charts of the sample to be classified, clarify the feature pattern of the sample, and determine whether the manifestation of each feature is a **typical feature** or **atypical feature** based on the heatmap (i.e., the corresponding color marking).

Combined with the extracted features of the sample to be classified and the provided list of features and patterns for different labels, analyze which label's features the sample features are more similar to and explain the reasons. The similarity judgment criterion is to analyze whether the feature pattern of the sample to be classified matches the pattern list of a certain label; if it matches, the sample is determined to belong to the same label. If an accurate result cannot be obtained based on the feature summary, or there is no fully matching pattern description in the feature summary, a comparative analysis can be conducted with reference samples to determine the most likely classification result.

The following is the core background information of the three types of charts:
- **Line Plot**: The line plot of each sample can provide classification-worthy features such as shape, rate of change, trend, peaks, and valleys. Since the Y-axis value range varies across samples, an auxiliary line y=0 is drawn in the chart to facilitate measuring the deviation degree of data points relative to y=0 and help analyze the magnitude relationship of peaks/valleys.
- **Time-Frequency Image**: High-energy regions in the time-frequency image correspond to peaks or valleys in the line plot that deviate from y=0 — the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. The time-frequency image also provides rich frequency information, which can decompose the signal into different frequency components. For example, if a certain peak/valley in the line plot has jitter, the time-frequency image will accurately capture it as a high-energy region distributed over a wider range of frequency components; the more severe the jitter, the more frequency components the high-energy region may cover, thereby reflecting the dispersion of frequency distribution.
- **Heatmap**: Generated based on the time-frequency image, it reflects the contribution of energy in different time regions and frequency components to the classification of the black-box model. Red regions indicate that the energy of the current time period and frequency component is a **typical feature** of the corresponding label; blue regions indicate that it is an **atypical feature**. The role of the heatmap is to directly associate the energy information of the time-frequency image with classification labels, helping identify which feature events are critical for classification.

<Analysis Methods and Rules>
The heatmap reflects the contribution of features in different regions to the classification of the black-box model. Therefore, the features in different regions may exhibit the following situations:
1. If a region's features are **predominantly marked in red** (allowing for a small amount of blue marking), it indicates that the black-box model considers the features in this region to be typical features of the current label;
2. If a region's features are **predominantly marked in blue** (allowing for a small amount of red marking), it indicates that the black-box model considers the features in this region to be atypical features of the current label, showing characteristics more similar to other labels;
3. If a region's features show a **mix of red and blue**, it indicates that the black-box model is confused about the features in this region, showing similar features that are difficult to distinguish from other labels.

Red and blue regions in the heatmap may be distributed in different time regions or different frequency components of the same time region. Therefore, the heatmap can not only reflect which time-region events are typical/atypical features, but also indicate which frequency components of the same event are typical/atypical features. When describing features, it is necessary to strictly determine whether the manifestation of a feature is a typical feature, an atypical feature, or an ambiguous feature that is difficult to distinguish.

<Task Requirements>
Please think step by step:
Step 1: Analyze the line plot, time-frequency image, and heatmap of the sample to extract as much valuable information as possible; summarize all features present in the sample, their manifestations (value range, amplitude range, shape, trend, volatility, etc.), the heatmap color types corresponding to these different feature manifestations, and determine whether these feature manifestations are typical features, atypical features, or ambiguous features that are difficult to distinguish. If necessary, frequency components can be divided and described (e.g., 0Hz row, adjacent upper frequency rows, mid-frequency region, high-frequency region, etc.).
Step 2: Summarize the feature pattern of the sample, i.e., the combination relationship between the features analyzed in Step 1 and their different manifestations. Analyze which features the sample consists of, what the manifestations of these features are, and their corresponding colors and feature types. If necessary, the sequence can be divided into meaningful phases (e.g., early stage, middle stage, late stage). (Example: The feature pattern of Sample X is: **excessive baseline deviation in the early stage (below -1) + stable amplitude in the late stage**; among them, the early-stage baseline is marked red in the heatmap, and the late-stage stable amplitude is marked blue.)
Step 3: Analyze whether the provided **list of label features and patterns** includes the feature pattern of the sample to be classified, and select the label with the highest matching degree as the final classification result.
Step 4: Before outputting the final result, conduct further analysis in combination with reference samples to recheck the accuracy of the thinking process in each step, the fairness of the scoring logic, and the consistency between the feature summary, reference sample information, and the classification result. After comprehensive and careful verification, output the most accurate classification result, and quantify the feature similarity between the sample to be classified and each label using an integer score from 1 to 4.

The description of each similarity level is as follows:
  + 1 point: **No similarity** — The sample to be classified has almost no correlation with the feature summary of the label, with no matching significant features or feature patterns; it shows significant differences in visual patterns from the reference samples of the corresponding label, and key feature points are inconsistent.
  + 2 points: **Low similarity** — There are 1-2 weakly correlated feature matches, but no complete feature pattern match; the conformity of feature manifestations is less than 50%. It has some similarity in basic shape to the reference samples, but there are obvious differences in key feature regions.
  + 3 points: **Moderate similarity** — There are multiple feature matches (≥3), with a basically matching feature pattern; the conformity of feature manifestations is between 50% and 80%. It has high similarity in key feature regions to the reference samples of the corresponding label, and the shape pattern is basically consistent.
  + 4 points: **High similarity** — The main features (≥80%) in the feature list are highly matched, with a completely matching feature pattern; the conformity of feature manifestations is >80%. It is highly similar in visual terms to the reference samples of the corresponding label, and key feature regions are almost identical.

<Output Requirements>
The output result must be in JSON format and strictly follow the following specifications:
{
    "result": ["label 0/1/……"],
    "score": [integer between 1 and 4],
    "rationale": ["your complete step-by-step analysis process"]
}
'''

prompt41 = '''
<Background>
You are an expert in time series data analysis, capable of analyzing **line plots** and **time-frequency images**. I will provide you with the following information:
- Line plot and time-frequency image of the sample to be classified;
- Classification results of the sample to be classified provided by three other assistants;
- The classification result and predicted probability value of the sample generated by the black-box model;
- Feature summaries of different labels, as well as potential feature patterns that different labels may have (combined features of different feature manifestations); among them, features marked red in the heatmap are typical features of the corresponding label, features marked blue are atypical features of the corresponding label, and features with a mixture of red and blue are similar features that are indistinguishable from other labels;
- Analysis processes of the sample provided by other assistants, focusing on exploring whether the sample to be classified is a boundary sample that causes misclassification by the black-box model.

Your task is to comprehensively analyze which label the sample to be classified should be classified into based on the provided information. Specifically, accurately extract all features from the line plot and time-frequency image of the sample to be classified, and clarify the feature pattern of the sample. Combined with the extracted features of the sample to be classified, the classification results of the three assistants, the logarithmic probability of the black-box model, and the provided feature and pattern lists of different labels, analyze which label's features the sample's features are more similar to and explain the reasons. Since the sample to be classified is identified as a boundary sample that may cause misclassification by the black-box model, it may exhibit features similar to those of other labels. When conducting comparisons, care should be taken not to regard misclassification features as direct classification basis.

The following is core background information about these two types of charts:
- **Line Plot**: The line plot of each sample provides classification-relevant features such as morphology, rate of change, trend, peaks, and valleys. The Y-axis value ranges of different samples vary, and an auxiliary line y=0 is plotted in the chart to facilitate measuring the deviation of data points relative to y=0. The amplitude and position of these features (e.g., peaks or valleys) are the core basis for analysis.
- **Time-Frequency Image**: High-energy regions in the time-frequency image correspond to peaks or valleys of the line plot that deviate from y=0—the greater the deviation from y=0, the higher the energy in the corresponding time window of the time-frequency image. The time-frequency image also provides rich frequency information, allowing signal decomposition into different frequency components: for example, if a peak/valley in the line plot exhibits jitter, the time-frequency image will accurately capture it as a high-energy region distributed across a wider range of frequency components; the more severe the jitter, the more frequency components the high-energy region may cover, thereby reflecting the degree of frequency distribution dispersion.

<Task Requirements>
Please think step by step:
Step 1: Extract the features, feature manifestations (value range, amplitude range, morphology, trend, volatility, etc.) and feature pattern of the sample to be classified, analyze which features constitute the sample and what their manifestations are. If necessary, the sequence can be divided into meaningful phases (e.g., early stage, middle stage, late stage).
Step 2: Combined with the content of other assistants exploring whether the sample to be classified is a boundary sample that causes misclassification by the black-box model, analyze which features in the sample may cause misclassification by the black-box model and are similar features indistinguishable from other labels.
Step 3: Match the features and patterns of the sample to be classified with the feature summary content to give a preliminary judgment on the sample to be classified; at the same time, combined with the results of Step 2, do not regard features that are prone to misclassification as direct judgment basis.
Step 4: Based on the preliminary judgment of the sample to be classified, analyze the classification results of the black-box model and the three assistants, and finally give the most reasonable classification result and alternative result in your opinion.
Step 5: Sort out all your thinking and reasoning processes, check whether there is any deviation from the task requirements; verify whether the optimal classification result is classified as an alternative result due to ignoring potential pattern features. After comprehensive verification, give the final and most accurate classification result.

<Output Requirements>
Field Descriptions:
- result field: The final classification result of the sample to be classified;
- model field: The classification result generated by the black-box model;
- rationale field: The complete step-by-step analysis process.

The output results must be in JSON format **and no additional content may be added**, and strictly comply with the following specifications:
{
    "result":["label 0/1/……"],
    "model":["label 0/1/……"],
    "rationale":["Your complete step-by-step analysis process"]
}
'''

prompt42 ='''
<Research Background>
As an expert in time series data analysis, you are capable of analyzing line plots, time-frequency images, and heatmaps. I will provide you with the following information:
- Line plot, time-frequency image, and heatmap of the sample to be classified;
- Feature summaries of different labels, as well as the feature patterns that different labels may possess (combined features of different feature manifestations);
- Classification results of the sample to be classified provided by three other assistants;
- Classification result and predicted probability value generated by the black-box model.

Your task is to judge the credibility of the black-box model's classification result based on the model's classification result, logarithmic probability value, and heatmap information. Specifically, observe whether the logarithmic probability value of the classification result is significantly higher than that of other labels; at the same time, combined with the information in the heatmap, observe the color-marked regions to analyze whether the sample has more typical features or atypical features. If the heatmap shows that the features of the sample are typical features, it indicates that the black-box model has a certain level of confidence in the classification result; if regions marked blue or mixed with red and blue are dominant, it indicates that the sample has some features that are similar and indistinguishable from those of other labels. Finally, combined with the feature summaries, the classification results of the three assistants, and your judgment on the black-box model's classification result, give the final classification result.

The following is the core background information about these three types of charts:
- **Line Plot**: The line plot of each sample can provide classification-relevant features such as morphology, rate of change, trend, peaks, and valleys. Since the Y-axis value ranges of different samples vary, an auxiliary line y=0 is plotted in the chart to facilitate measuring the deviation of data points relative to y=0, which helps analyze the magnitude relationship of peaks and valleys.
- **Time-Frequency Image**: High-energy regions in the time-frequency image correspond to the peaks or valleys of the line plot that deviate from y=0—the greater the deviation from y=0, the higher the energy in the corresponding time region of the time-frequency image. The time-frequency image also provides rich frequency information and can decompose the signal into different frequency components: for example, if a peak or valley in the line plot has jitters, the time-frequency image will accurately capture it as a high-energy region distributed over a wider range of frequency components; the more severe the jitters, the more frequency components the high-energy region may cover, thereby reflecting the degree of dispersion of the frequency distribution.
- **Heatmap**: Generated based on the time-frequency image, it reflects the contribution of energy in different time regions and frequency components to the classification of the black-box model. Red regions indicate that the energy of the current time region and frequency component is a typical feature of the corresponding label; blue regions indicate that it is an atypical feature. The function of the heatmap is to directly correlate the energy information of the time-frequency image with classification labels, helping to determine which feature events are crucial for classification.

<Analysis Methods and Rules>
The heatmap reflects the contribution of features in different regions to the classification of the black-box model. Therefore, features in different regions may present the following scenarios:
1. If red markings dominate a region (with a small amount of blue markings allowed), it means the black-box model regards the features of this region as typical features of the current label;
2. If blue markings dominate a region (with a small amount of red markings allowed), it means the black-box model regards the features of this region as atypical features of the current label, which show more similarities to the features of other labels;
3. If a region has a mixture of red and blue markings, it means the black-box model is confused about the features of this region, which show indistinct similarities to those of other labels.

The red and blue regions in the heatmap may be distributed in different time regions or across different frequency components within the same time region. Therefore, the heatmap can not only reflect which time-region events are typical/atypical features, but also indicate which frequency components of the same event are typical/atypical features. When describing features, it is necessary to strictly judge whether the manifestation of a feature is a typical feature, an atypical feature, or an indistinct ambiguous feature.

<Task Requirements>
Please think step by step:
Step 1: Analyze the line plot, time-frequency image, and heatmap of the sample to extract as much valuable information as possible; summarize all features appearing in the sample and their manifestations (value range, amplitude range, morphology, trend, volatility, etc.), the heatmap color types corresponding to different manifestations of these features, and judge whether these feature manifestations are typical features, atypical features, or indistinct ambiguous features. If necessary, frequency components can be divided and described (e.g., 0Hz row, adjacent upper frequency rows, intermediate frequency region, high frequency region, etc.).
Step 2: Evaluate the model’s classification result and logits by combining heatmap information, assess the confidence level of the model’s prediction, and determine whether the sample to be classified is a boundary sample that is likely to cause misclassification by the black-box model or a typical sample of a certain label.
Step 3: Combined with the feature summaries of labels, and based on the classification results of the black-box model and the three assistants, select the most reasonable classification result and your alternative result from the four results of the black-box model and the three assistants.
Step 4: Reorganize all your thinking and reasoning processes, check whether there is any deviation from the task requirements; verify whether the optimal classification result is classified as an alternative result due to ignoring potential pattern features; after comprehensive verification, give the final and most accurate classification result.

<Output Requirements>
Field Descriptions:
- result field: The final classification result of the sample to be classified;
- model field: The classification result generated by the black-box model;
- rationale field: The complete step-by-step analysis process.

The output result must be in JSON format **without any additional content**, and the format must strictly comply with the following specifications:
{
    "result":["label 0/1/……"],
    "model":["label 0/1/……"],
    "rationale":["your complete step-by-step analysis process"]
}
'''