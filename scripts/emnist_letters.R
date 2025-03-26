## Emnist, just first ten letters 

lt<-read_csv("emnist-letters-train.csv")%>%
  clean_names()

names(lt)<-c("label",
             paste0("pix_",1:784))

lt<-lt%>% 
  filter(label %in% 1:10)


write_csv(lt,file="emnist_first_ten.csv")
