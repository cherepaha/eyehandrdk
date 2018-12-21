read.files <- function(files){
  firstRun = T
  for(file.ix in 1:length(files)){
    if(firstRun == F){
      hold.data = read.table(paste(data.folder, files[file.ix], sep = "/"), header = T)  
      alldata = rbind(alldata, hold.data)
      rm(hold.data)
    } else alldata = read.table(paste(data.folder, files[file.ix], sep = "/"), header = T)  
    # firstRun
    firstRun = F
  } # file.ix
  return(alldata)
}