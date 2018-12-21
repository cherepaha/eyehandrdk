                ##############################################
                ###  THIS SCRIPT DOWNSAMPLES LARGE FILES   ### 
                ##############################################


# Pulling in samples from very large files function
sample.file <- function(filename, rowsample = T, rowsample.skip = 100, 
                        numsamples = 100){
  
  # use numsamples == "All" to sample whole file
  
  # rowsample = T, # sample by row?
  # rowsample.skip = 100, # if rowsample is T, then is the number of rows between samples
  
  # TO DO: 
  # timesample = F, # sample by time?
  # timesample.rate = 100 # if timesample = T, this is sampling rate in Hz
  if(numsamples == "All"){
    print("Assessing file size, please wait")
    testcon <- file(files[1],open="r")
    readsizeof <- 20000
    nooflines <- 0
    ( while((linesread <- length(readLines(testcon,readsizeof))) > 0 )
      nooflines <- nooflines+linesread )
    close(testcon)
    nooflines
    print(paste("Number of lines =",nooflines))
  } # numsamples
  
  
  # initalising vectors
  hold.row = 0
  all.rows = 0
  firstRun = T
  
  # get data headers
  data.headers = readLines(filename, n = 1)
  data.headers = unlist(strsplit(data.headers, split = "\t"))
  
  if(rowsample == T) skip = rowsample.skip
  # TO DO: if(timesample == T) #skip = timesample.rate # wrk this out based on time subtraction TBD
  print(paste("Please Wait. Number of samples =", numsamples))
  is.wholenumber <- function(x, tol = .Machine$double.eps^0.5)  abs(x - round(x)) < tol
  
  for(sample in 1:numsamples){
    hold.row = scan(filename, nlines = 1, skip = ((sample-1)*skip) +1, quiet = T)
    
    if(is.wholenumber(sample/100)) print(paste("Running sample no.", sample))
    
    if(firstRun == F) all.rows = c(all.rows, hold.row)
    else all.rows = hold.row
    
    firstRun = F
  }
  
  
  df = data.frame(matrix(all.rows, ncol = 7, byrow = T))
  colnames(df) = data.headers
  return(df)
  
}

###------------ CALCULATING THE SAMPLING RATE ----------------------
test = sample.file(files[1], rowsample.skip = 1, numsamples = 20) # change the index of files[] for each of the subsequent files
sample.rate = 1000/((test$timestamp[20] - test$timestamp[1])/20)
rows.toskip = round(sample.rate/60,0)

###----------------------------------------------------------------

## numsamples = nooflines/rows.toskip
## numsecs = numsamples/60
## numsecs/5

## test = scan(, nlines = 10, skip = 1)
## parameters of call
## filename = "6590_Nov_03_2016_15_16_22_consequence_image.txt"
## skip = 100 # work this out from sampling rate
## nsamples = 100


####################### USING THE FUNCTION ############################
###------------ change folder in accordance (e.g., santi.folder) ----------------------
# doh.folder = "/Volumes/SANTI/Exp I (LAB)BackUp/data"
# setwd(doh.folder) 
santi.folder = "E:/Santi/Exp I (LAB version)/data/week1"
setwd(santi.folder) 

###------------ Slecting the files from the folder ----------------------
filesize = file.size(dir())
files = list.files()
files = files[filesize>10^7] # files over 100 mbytes

### ------- Before anything else, run the coded function ---------------

###------------ calculate sampling rate for each file -------------------
# File [1]
test = sample.file(files[1], rowsample.skip = 1, numsamples = 20) 
sample.rate = 1000/((test$timestamp[20] - test$timestamp[1])/20)
rows.toskip = round(sample.rate/60,0)

# File [2]
test = sample.file(files[2], rowsample.skip = 1, numsamples = 20) 
sample.rate = 1000/((test$timestamp[20] - test$timestamp[1])/20)
rows.toskip = round(sample.rate/60,0)

# File [...]
test = sample.file(files[12], rowsample.skip = 1, numsamples = 20) 
sample.rate = 1000/((test$timestamp[20] - test$timestamp[1])/20)
rows.toskip = round(sample.rate/60,0)


###------------ downsampling each of the files [as indicated by the files variable in your environment or length(files) ]---------------------
## First, determine how many obs. a normal file has, so this number can be passed to the "numsamples =" argument 
normalFile = read.csv("F:/Exp I (LAB)BackUp/data/2436_Nov_24_2016_15_13_37_consequence_image.txt")

## Second, get the sampling rate for each of the files (and run it!)
rows.toskip1 = 800
rows.toskip2 = 886
rows.toskip3 = 971
rows.toskip4 = 909
rows.toskip5 = 796
rows.toskip6 = 849
rows.toskip7 = 961
rows.toskip8 = 960
rows.toskip9 = 621
rows.toskip10 = 876
rows.toskip11 = 810
rows.toskip12 = 880
rows.toskip13 = 880

# First file 
downSampledFile.1 = sample.file(files[1], rowsample.skip = rows.toskip1, numsamples = 20000) 
# Second  file
downSampledFile.2 = sample.file(files[2], rowsample.skip = rows.toskip2, numsamples = 20000) 
downSampledFile.2 = sample.file(files[3], rowsample.skip = rows.toskip3, numsamples = 20000) 
downSampledFile.5 = sample.file(files[4], rowsample.skip = rows.toskip4, numsamples = 20000) 
downSampledFile.6 = sample.file(files[5], rowsample.skip = rows.toskip5, numsamples = 20000) 
downSampledFile.7 = sample.file(files[6], rowsample.skip = rows.toskip6, numsamples = 20000) 
downSampledFile.8 = sample.file(files[7], rowsample.skip = rows.toskip7, numsamples = 20000) 
downSampledFile.9 = sample.file(files[8], rowsample.skip = rows.toskip8, numsamples = 20000) 
downSampledFile.9 = sample.file(files[9], rowsample.skip = rows.toskip9, numsamples = 20000) 
downSampledFile.10 = sample.file(files[10], rowsample.skip = rows.toskip10, numsamples = 20000) 
downSampledFile.11 = sample.file(files[11], rowsample.skip = rows.toskip11, numsamples = 20000) 
downSampledFile.12 = sample.file(files[12], rowsample.skip = rows.toskip12, numsamples = 20000) 
downSampledFile.13 = sample.file(files[13], rowsample.skip = rows.toskip13, numsamples = 20000) 

###------------ saving the downsampled files -------------------
# files[1]
write.table(downSampledFile.1, file = "E:/Santi/downSampled/1081_Nov_01_2016_10_54_14_consequence_image.txt", sep= "\t", row.names = FALSE)

# files[2]
write.table(downSampledFile.2, file = "E:/Santi/downSampled/1860_Nov_04_2016_17_05_32_consequence_image.txt", sep= "\t", row.names = FALSE)

# files[3]
write.table(downSampledFile.3, file = "E:/Santi/downSampled/2456_Nov_03_2016_11_04_28_consequence_image.txt", sep= "\t", row.names = FALSE)

# files[4]
write.table(downSampledFile.4, file = "E:/Santi/downSampled/3236_Nov_04_2016_14_11_00_consequence_image.txt", sep= "\t", row.names = FALSE)

# files[5]
write.table(downSampledFile.5, file = "E:/Santi/downSampled/3896_Nov_03_2016_12_24_33_consequence_image.txt", sep= "\t", row.names = FALSE)

# files[6]
write.table(downSampledFile.6, file = "E:/Santi/downSampled/5773_Nov_03_2016_15_56_24_consequence_image.txt", sep= "\t", row.names = FALSE)

# files[7]
write.table(downSampledFile.7, file = "E:/Santi/downSampled/6103_Nov_03_2016_12_54_10_consequence_image.txt", sep= "\t", row.names = FALSE)

# files[8]
write.table(downSampledFile.8, file = "E:/Santi/downSampled/6590_Nov_03_2016_15_16_22_consequence_image.txt", sep= "\t", row.names = FALSE)

# files[9]
write.table(downSampledFile.9, file = "E:/Santi/downSampled/7135_Nov_03_2016_17_23_21_consequence_image.txt", sep= "\t", row.names = FALSE)

# files[10]
write.table(downSampledFile.10, file = "E:/Santi/downSampled/7248_Nov_03_2016_10_19_51_consequence_image.txt", sep= "\t", row.names = FALSE)

# files[11]
write.table(downSampledFile.11, file = "E:/Santi/downSampled/7313_Nov_01_2016_17_53_52_consequence_image.txt", sep= "\t", row.names = FALSE)

# files[12]
write.table(downSampledFile.12, file = "E:/Santi/downSampled/7835_Nov_04_2016_15_15_33_consequence_image.txt", sep= "\t", row.names = FALSE)

# files[13]
write.table(downSampledFile.13, file = "E:/Santi/downSampled/9945_Nov_02_2016_12_19_44_consequence_image.txt", sep= "\t", row.names = FALSE)











