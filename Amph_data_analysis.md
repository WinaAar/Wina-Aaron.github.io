---
title: "Amphibian data analysis"
author: "Wina Aaron"
format: 
  html:
    self-contained: true
editor: visual
---

```{r setup}
#| include: false
library(tidyverse)
library(readr)
```

## Summary

This project option is based on a data set of observations called ["Mohonk Preserve Amphibian and Water Quality Monitoring Dataset at 11 Vernal Pools from 1931-Present"](https://portal.edirepository.org/nis/mapbrowse?scope=edi&identifier=398&revision=7). The metadata (description of data) are provided [here](https://portal.edirepository.org/nis/metadataviewer?packageid=edi.398.7); click on "Data entities" tab for explanations of each variable. There are three different data frames in this data set, which are loaded by running the code chunk provided below:

-   `amph_obs` contains observations of different amphibian species at different locations;

-   `loc_weather` contains measurements of ambient conditions at different lovations, such as precipitation, air and water temperature;

-   `locations` contains geographical coordinates of the different location.

```{r}
amph_obs <- read_csv("https://raw.githubusercontent.com/dkon1/quant_life_quarto/main/data/project%20data/amph_obs.csv")
loc_weather <- read_csv("https://raw.githubusercontent.com/dkon1/quant_life_quarto/main/data/project%20data/loc_weather.csv")
locations <- read_csv("https://raw.githubusercontent.com/dkon1/quant_life_quarto/main/data/project%20data/locations.csv")
spec(amph_obs)
spec(loc_weather)
spec(locations)
```

```{r}
head(amph_obs)
```

```{r}
head(loc_weather)

```

### Reading:

1.  [Dataset description](https://portal.edirepository.org/nis/mapbrowse?scope=edi&identifier=398&revision=7) - go to View Full Metadata and click on Data Entities to see full description of each variable
2.  [Vernal pool amphibian breeding ecology monitoring from 1931 to present: A harmonised historical and ongoing observational ecology dataset](https://bdj.pensoft.net/article/50121/) - published article that uses this data set
3.  [R for Data Science](https://r4ds.had.co.nz/data-visualisation.html) - best overall reference for tidyverse and ggplot; see chapters 3, 5, and 7
4.  [ggplot tutorial](https://uc-r.github.io/ggplot_intro) - for another introduction to ggplot
5.  [Lisa Lendway's tidyverse + ggplot intro](https://ggplot-dplyr-intro.netlify.app/) - another good resource

The project consists of four steps:

### Step 1: Data exploration and project proposal

The function `glimpse` prints out a short description of each variable in the data set:

```{r}
glimpse(amph_obs)
```

```{r}
amph_obs |> count(CommonName)
```

```{r}
amph_obs |>  # take the data set (tibble)
  drop_na(Live_n) |>  # drop observations that have missing values of Live_n
  group_by(CommonName, Location) |>  # divide observations by species and section
  summarise(count = n(), mean_live = mean(Live_n), sd_live = sd(Live_n))
```

Visualize individual variables using histograms or density plots, for example here is a histogram of the variable length_1_mm:

```{r}
amph_obs |>  # take the data set (tibble)
  drop_na(Live_n) |> # drop observations that have missing values of Live_n
  ggplot() + aes(x=Live_n) + # variable to plot
  geom_histogram()# histogram of variable

```

visualize two variables together to visually assess their relationship. For example, here is a visualization of the distribution of total of different species for different locations:

```{r}
amph_obs |>  # take the data set (tibble)
  drop_na(Live_n) |>  # drop observations that have missing values of Live_n
  ggplot() + aes(x=Location, y=Live_n) + # selection of explanatory and response variables 
  geom_boxplot(aes(fill = Location)) + coord_flip()  # boxplot in horizontal orientation

```

Goal for this Stage of the Project :

1.  State a simple statistical question of your choice (e.g. is one variable independent of another? or what is the best-fit `m`odel for one variable as a function of another?) and explain it in biological terms;

    I wanted to compare the relationship between the groups that are of concern and not of concern and whether or not there exists significant differences in the egg mass collected of the most abundant species in the respective groups.

    (In short, is EggMass dependent on Ny_concern?)

2.  State the method you choose to answer this question statistically (e.g. use a chi-squared test, perform linear regression);

    I will be using the summary table I obtain through a linear regression model to observe if any statistic significant exists to help me answer my question about the relationship between NY_concern and EggMass_n.

3.  State the visualization you will use to illustrate this (e.g. scatterplot, boxplot);

    I will be using boxplots in this project to observe the difference between species of concern and non-concern. I will also be looking at a histogram of my response variable to see if transformation is needed to normalize the data.

4.  Briefly explain the assumptions of the method and how you will assess whether they are not violated (e.g. examine plot of residuals);

    I will be to using a qqnorm plot to check the normality of my linear model and a fitted vals v residuals plot to observe the variance of model to check that it is constant.

5.  Based on the results of your initial investigation, you may come up with a more interesting question that involves a multiple variables or a more complex relationship; for example, use the weather conditions data to build a model of how amphibian populations are affected by rainfall or test a hypothesis about one or more species. You don't need to include this in your proposal, but you may list some possible ideas if you want feedback!

    For the complex question I will be joining the weather and species tables and fit a multi-linear regression in order to see if the predictors are signifcant in predicting the response. As my predictors I have chosen to look at water_temp,air_temp, live_n, PDP, water_depth, and Ny_concern. I will be creating histograms and scatter plots to visualize the relationship the predictors have with the response and each other.

## Step 2: Data cleaning and filtering

Performing data cleaning and manipulation in the code block below and assign the result to a new data frame you can use. Describe your choices and why you made them in the space below:

```{r}
library(ggplot2)
library(dplyr)

#removing some repettive info/unneeded info 
new_amph = amph_obs[, c("CommonName", "NY_Concern", "Location",  "EggMass_n", 'Live_n')]
head(new_amph)
```

```{r}
#checking for nuls in ordr in order to impute, too many nulls to simply drop
null_sums <- colSums(is.na(new_amph))
print(null_sums)


```

```{r}

#dropping the mising values in Live_n
clean = new_amph %>%
  drop_na(EggMass_n)
```

```{r}
#checking nulls are gone

sum(is.na(clean$EggMass_n))

```

```{r}
#looking at the species of concern and nonconcern with the most observtions
concern_spec = subset(clean, NY_Concern ==1)
nonconcern_spec = subset(clean, NY_Concern == 0)
head(concern_spec)
head(nonconcern_spec)


```

```{r}
#finding which species is most abundant in both subset datasets
most_concern = concern_spec %>%
  group_by(CommonName) %>%
  summarise(total = sum(Live_n, na.rm = TRUE)) %>%
  arrange(desc(total)) %>%
  slice(1) %>%
  pull(CommonName)
cat('Most abundant specie of concern:', most_concern)

most_nonconcern = nonconcern_spec %>%
  group_by(CommonName) %>%
  summarise(total = sum(Live_n, na.rm = TRUE)) %>%
  arrange(desc(total)) %>%
  slice(1) %>%
  pull(CommonName)
cat('Most abundant specie of nonconcern:', most_nonconcern)

```

Analysis:

I decided to create a subset of the original dataset to look at solely the variables I was interested in looking at for my model and visualizations. I then checked the sum of missing values in each one of the variables and noticed that Egg_mass has 715 missing values, so I dropped those rows in order to minimize error. I then split the dataframe into two more subsets and I separated them by NY_Concern values to get one that had all the species of concern and one that contained all the species that are not of concern. I found the sums of live_n in both the dataframes and grouped them by the common name in order to find the most abundant specie of concern and non concern. (Following sentence is in the following portion) Finally, I concatenated the two data sets and filtered them out to only include the information of the two common names I found as being the most abundant for the binary choices of NY_Concern (Jefferson Salamander and Wood Frog).

## Step 3: Data analysis for a simple question

Analysis and Visualization:

First, visualize the relationship that you are exploring, usually between an explanatory or response variable, or show distributions of values in different conditions.

```{r}
spec_select <- clean %>%
  filter(CommonName %in% c(most_concern, most_nonconcern))
head(spec_select)
```

```{r}

ggplot(spec_select,aes(x=as.factor(NY_Concern) , y = EggMass_n, fill = as.factor(NY_Concern))) +geom_boxplot() + labs(x = 'NY Concern', y = 'Egg Mass observations', fill = 'NY Concern') 

```

```{r}
#want to look at the distrubution of the repsonse variable
ggplot(spec_select, aes(x = EggMass_n)) + geom_histogram(binwidth = 100, fill = 'red', color = 'black')
```

Perform statistical analysis or modeling, or any other work to answer the question you posed:

```{r}
library(broom)
```

```{r}
#the histogram above shows the data is heavily right skewed. decided to apply log transformation to help with normality and the variance.
epsilon <- 0.001 
spec_select$EggMass_n_log <- log(spec_select$EggMass_n + epsilon)

summary(spec_select$EggMass_n_log)
head(spec_select)

spec_select$NY_Concern = as.factor(spec_select$NY_Concern)
lmlog = lm(EggMass_n_log ~ NY_Concern, data = spec_select)
summary(lmlog)
```

All models and methods are based on assumptions. Explain what they are and perform a calculation or visualization to make sure there are no violations in your data set. Note that if the assumptions of your method or model are substantially violated, you'll need to address this issue; either by choosing another method, or rethinking your variable selection, removing outliers if it's appropriate, etc.

```{r}
#checking normaltiy 
qqnorm(lmlog$residuals)
qqline(lmlog$residuals)

#checking variance 
plot(fitted(lmlog), residuals(lmlog))
```

YOUR EXPLANATIONS HERE:

I utilized a linear regression model for my research. Linear regression assumes normality, constant variance, and independence. I initially fit the model with the original values of the variables and checked these assumptions by creating a qqnorm plot and fitted vs. residuals plot . I found that the data was not normal and violated the constant variance assumption; therefore, I decided to apply log transformation to my response. I checked variance and normality again using the log transformed model, and the variance improved and the normality plot shows that the plot is light-tailed, which is not detrimental in terms of using linear regression modeling for the data.

Write a short summary of what you did: explain what it means for the question, comment on what you learned and how certain you are of the conclusions. This does not need to be a long essay!

YOUR WORDS HERE:

My initial question for the simple question portion of the project was whether NY_Concern had an impact on the egg_mass. I checked for violations of linear assumptions, log-transformed my data to make it better suitable for linear regression, and obtained the summary table using my model. The summary table provides us with the coefficient estimates and p-values of the predictors. Using the information from the summary table, I learned that the NY_Concern variable is highly statistically significant when predicting the EggMass_n_log. The -4.67 coefficient of NY_Concern indicates that when NY_concern is 1, the expected value of EggMass_n is lower compared to when NY_concern is 0, holding all other variables constant. I feel the reuslts make perfect sense in the context of the data. Species of concern are producing less eggs than those that are of none concern.

# Step 4: More complex question

Prepare and clean the data set for the question you asked:

```{r}
#joining the tables on their common variable location
combined = inner_join(amph_obs, loc_weather, by = 'Location', relationship = 'many-to-many')
head(combined)

#dropping columns i dont need for my research
new_combined <- combined %>%
  dplyr::select(-ScientificName,-ITIS_TSN, -Authorship, -Sample_Date.x, -Sample_Date.y, -ChorusCode, -ChorusCount, -Juv_n, -Sperm_n, -TadLarv_n, -Dead_n, -AmplectantPairs_N, -pH, -Water_Level, -Visibility_Imp, -Surface_Ice,-Surface_Veg , -Odor, -Turbidity_NTU, -Wind_Code, -Sky_Code, -Shrimp, -Snails, -Chloride, -Nitrate, -SurfaceVeg_Spp, -Conductivity, -DO)
head(new_combined)
```

```{r}
null_sums <- colSums(is.na(new_combined))
print(null_sums)
```

```{r}
clean_com = new_combined %>%
  drop_na(EggMass_n, Air_Temp_C, PDP, Live_n, Water_Temp_C, Water_Depth )

colSums(is.na(clean_com))
head(clean_com)
```

```{r}
#looking at the species of concern and nonconcern with the most observtions
concern_spec1 = subset(clean_com, NY_Concern ==1)
nonconcern_spec1 = subset(clean_com, NY_Concern == 0)
head(concern_spec1)
head(nonconcern_spec1)
most_concern1 = concern_spec1 %>%
  group_by(CommonName) %>%
  summarise(total = sum(Live_n, na.rm = TRUE)) %>%
  arrange(desc(total)) %>%
  slice(1) %>%
  pull(CommonName)
cat('Most abundant specie of concern:', most_concern)

most_nonconcern1 = nonconcern_spec1 %>%
  group_by(CommonName) %>%
  summarise(total = sum(Live_n, na.rm = TRUE)) %>%
  arrange(desc(total)) %>%
  slice(1) %>%
  pull(CommonName)
cat('Most abundant specie of nonconcern:', most_nonconcern)
```

```{r}
spec_select1 <- clean_com %>%
  filter(CommonName %in% c(most_concern1, most_nonconcern1))
spec_select1
```

Make a visualization for the relationship or question you wish to explore:

```{r}
#relationship beteern egg mass and ekevation
plot(spec_select1$Air_Temp_C, spec_select1$EggMass_n, main="Air Temp Vs. Egg Mass", xlab="Air temp", ylab="Egg Mass")
```

```{r}
ggplot(spec_select1,aes(x=as.factor(PDP) , y = EggMass_n, fill = as.factor(PDP))) +geom_boxplot() + labs(x = 'Presence/absense of Previous Day Percp.', y = 'Egg Mass observations', fill = 'PDP', title= 'Percipitation V Egg Mass') 

```

```{r}
ggplot(spec_select1,aes(x= Location , y = EggMass_n, fill = Location)) +geom_boxplot(width = .7) + labs(x = 'Locations', y = 'Egg Mass observations', fill = 'Location') +coord_flip()
```

State the assumptions or foundations of the method you chose and explain how you tested them:

YOUR WORDS HERE:

I chose multi-linear regression modeling for my more complex method, and as stated above, linear regression assumes normality, independence, and constant variance. I have used qqnorm plots, fitted v residual plots, and VIF. The qqnorm demonstrates how normal my model residuals are, the fitted plot shows the variance of the model, and VIF will show me if any of the predictors are multicollinear impacting independence. I decided to log transform my response again. When I fit the model without transforming it first, the qqnorm plot showed that my data was right skewed and that the constant variance was violated. Post- transformation, my data is light-tailed again, and as stated in the first part this indicates that my data isn't exactly normal but it's not very far from it. The variance also shows that it improves and becomes more dispersed although it is still not perfectly constant. The adjusted model has removed the predictors that indicate extreme multicollinearity in order to maintain only the independent predictors in my model. It has also been log transformed in order to make it more normal and more constant in terms of the variance.

Perform the analysis to answer the question or model a relationship.

```{r}
library(car)
spec_select1$NY_Concern = as.factor(spec_select1$NY_Concern)
spec_select1$PDP = as.factor(spec_select1$PDP)

com_model = lm(EggMass_n ~ NY_Concern + Location + Live_n + PDP + Water_Temp_C  +Water_Depth , data = spec_select1 )
vif(com_model)


#checking normaltiy 
qqnorm(com_model$residuals)
qqline(com_model$residuals)

#checking variance 
plot(fitted(com_model), residuals(com_model))
```

```{r}
#removed air_temp and water_depth from my model because they had extremley high multicollinearity
spec_select1$log_EggMass_n <- log(spec_select1$EggMass_n + .001)

spec_select1$NY_Concern = as.factor(spec_select1$NY_Concern)
spec_select1$PDP = as.factor(spec_select1$PDP)

com_model_adjusted = lm(log_EggMass_n ~ NY_Concern + Location + Live_n + PDP + Water_Temp_C, data = spec_select1 )
vif(com_model_adjusted)

summary(com_model_adjusted)
#checking normaltiy 
qqnorm(com_model_adjusted$residuals)
qqline(com_model_adjusted$residuals)

#checking variance 
plot(fitted(com_model_adjusted), residuals(com_model_adjusted))
```

Write a short summary of what you did: explain what it means for the question, comment on what you learned and how certain you are of the conclusions. This does not need to be a long essay!

YOUR WORDS HERE:

I wanted to test whether the addition of Location, water_temp_c, and presence or absence of precipitation of the previous day (PDP) would have significant impact on predicting egg mass production. I cleaned my data, applied transformations and dropped predictors in order to combat violations of linear regression assumptions and obtain the most accurate results. I obtained the summary table of my adjusted model and looked at the p-values of the predictors. I learned how significant location was in terms of predicting egg mass. I also learned that PDP1 and Water_temp are do not hold significance in predicting egg mass, which I found interesting. The fact that majority of the locations were significant but the water temp and pdp were 1 makes me question my certainty for my conclusions. I would expect the water temp to be associated with location to have an impact on egg mass, but there are also a lot of confounding variables as to why locations are incredibly significant while water temp and the binary classification on whether percipitation occurred the day before are not significant in predicting egg mass.
