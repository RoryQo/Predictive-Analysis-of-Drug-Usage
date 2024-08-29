Lab 10: Unsupervised Machine Learning
================
Rory Quinlan

``` r
# Load Data
library(tidyverse)
library(factoextra)
library(cluster)

country = read_csv("Country-data.csv",show_col_types = FALSE)
```

#### <span style="color:light light blue"> Calculate proportions of variance explained for all of the 9 principal components. Show the vector of proportions.) </span>

``` r
# Take country deselect country
# Then do principal component analysis
country_pc = country %>% 
  select(-country) %>% 
  prcomp(scale=TRUE)

# Find variance explained by taking sdev^2 for each component
# Then dividing that by the sum of the variance
PRVar<- country_pc$sdev^2
PVE<- PRVar[1:9]/sum(PRVar)
PVE
```

    ## [1] 0.459517398 0.171816257 0.130042589 0.110531618 0.073402114 0.024842347
    ## [7] 0.012604304 0.009812817 0.007430556

#### <span style="color:light light blue"> Create the scree plot. Where is the **elbow**? </span>

``` r
# Create data frame to plot with PVE (above) and vec 1:9
PC=1:9
data=data.frame(PC, PVE)

# Plot pve of components
ggplot(data=data, aes(x=PC, y=PVE))+
  geom_line(color="navy")+
  geom_point(aes(x=6,y=0.023127004),cex=5,color="orange",alpha=0.3)+
  geom_point(color="red",cex=2)+
  labs(title="Proportion of Variance Explained", x="Principal Component",y="pve")+
  scale_x_continuous(breaks = 1:9)
```

![](Runsupervised-Learning-Lab_files/figure-gfm/unnamed-chunk-3-1.png)<!-- -->

-   The elbow is at the 6th pc data point. This is where the following
    data points begin leveling off significantly.

#### <span style="color:light blue"> Create the plot for the cumulative proportions of variance explained. Put the horizontal reference line at `y=0.9`.) </span>

``` r
ggplot(data=data, aes(x=PC, y=cumsum(PVE)))+
  geom_hline(aes(yintercept=0.9),lty=2,color="purple",linewidth=1, alpha=0.5)+
  geom_line(color="navy")+
  geom_point(color="red",cex=2)+
  labs(title="Cumulative Proportion of Variance Explained", 
       x="Principal Component",
       y="cumulative pve")+
  scale_x_continuous(breaks = 1:9)
```

![](Runsupervised-Learning-Lab_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

#### <span style="color:light blue"> Extract the first 4 principal components from `country_pc$x` and use them to find the optimal number of `k`. </span>

``` r
# Deselect country
country_s = scale(country[,-1])
# Select the first 4 principal components
optcluster <- country_pc$x %>% as_tibble() %>% select(1:4)

# Find the optimum number of k
fviz_nbclust(optcluster, kmeans, method = "gap_stat")
```

![](Runsupervised-Learning-Lab_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

#### <span style="color:light blue"> How many observations were clustered differently by the kmeans and the PAM methods? </span>

``` r
km_mod = kmeans(country_s, centers=3)
pam_mod = pam(country_s, 3)
sum(pam_mod$clustering != km_mod$cluster)
```

    ## [1] 88

#### <span style="color:light blue"> Create a boxplot to compare the GDP per capita (`gdpp`) for the three clusters: </span>

``` r
country_pam = country %>% mutate(cluster=factor(pam_mod$cluster))
boxplot(gdpp~cluster,data=country_pam, horizontal=T)
```

![](Runsupervised-Learning-Lab_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

#### <span style="color:light blue"> Calculate the median life expectancy for the three clusters. </span>

``` r
library(dplyr)

sumardata<-country_pam %>%
  group_by(cluster) %>%
  summarise(across(.fns = median))
sumardata$life_expec
```

    ## [1] 60.4 74.1 80.4

-   Cluster 1 life expectancy is 60.4 years
-   Cluster 2 life expectancy is 74.1 years
-   Cluster 3 life expectancy is 80.4 years