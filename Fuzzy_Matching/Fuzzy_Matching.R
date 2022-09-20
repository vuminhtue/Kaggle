rm(list=ls())
setwd('/work/users/tuev/FuzzyMatching/')
dfsp500 <- read.csv("input/SP500.csv")
dfft500 <- read.csv("input/Fortune500_2021.csv")
head(dfsp500)
head(dfft500)

m1 <- adist(dfsp500$Name,dfft500$Name)
ind1 <- apply(m1,1,which.min)
dfft500[ind1,]
out1 <- data.frame(dfsp500$Symbol,dfsp500$Name,dfsp500$Sector,dfsp500$Price,
                     dfft500$Name[ind1],dfft500$Revenue[ind1])
write.csv(out1,"output/adist.csv",row.names = FALSE)

# Other 
library(fuzzyjoin)
library(dplyr)
out2 <- stringdist_join(dfsp500,dfft500,
                by="Name",
                mode="left",
                method="jw",
                max_dist=40,
                distance_col='dist') %>%
  group_by(Name.x) %>%
  slice_min(order_by=dist,n=1)
write.csv(out2,"output/strdist_jw.csv",row.names = FALSE)
