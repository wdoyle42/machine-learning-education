library(janitor)
library(tidyverse)
library(tidymodels)

st<-read_csv("studentAssessment.csv")

si<-read_csv("studentInfo.csv")

sta<-read_csv("studentAssessment.csv")

stvle<-read_csv("studentVle.csv")

vle<-read_csv("vle.csv")

reg<-read_csv("studentRegistration.csv")

assess<-read_csv("assessments.csv")


## join vle and student vle

stvle<-stvle%>%left_join(vle)


## Aggregate clicks by activity type, class code and term
stvle<-stvle%>%group_by(id_student,code_module,code_presentation,activity_type)%>%
  summarize(clicks=sum(sum_click))

## Add information to student information
si<-si%>%left_join(stvle)

## Recode outcome

ou<-si%>%mutate(result=fct_collapse(as_factor(final_result),
                             passed=c("Pass","Distinction"),
                             not_passed=c("Fail","Withdrawn")))

## One variable per sum clicks per activity
ou<-ou%>%pivot_wider(names_from=activity_type,values_from = clicks)

## Messed up level shhhhhh
ou<-ou%>%select(-("NA"))

## q marks=NA
ou<-ou%>%
  mutate(across(where(is.character), ~na_if(., "?")))

write_csv(ou,file="oulad.csv")
