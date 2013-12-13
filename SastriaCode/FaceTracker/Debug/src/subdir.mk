################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/BBoxAtFrame.cpp \
../src/CalcHistPositive.cpp \
../src/DataSharing.cpp \
../src/DetectionToRecognition.cpp \
../src/FacePosition.cpp \
../src/FacesList.cpp \
../src/FerNNClassifier.cpp \
../src/FramesList.cpp \
../src/IntegralImage.cpp \
../src/LKTracker.cpp \
../src/PositionList.cpp \
../src/SingleFrame.cpp \
../src/TLD.cpp \
../src/ValidateDetection.cpp \
../src/run_tld.cpp \
../src/tld_utils.cpp 

OBJS += \
./src/BBoxAtFrame.o \
./src/CalcHistPositive.o \
./src/DataSharing.o \
./src/DetectionToRecognition.o \
./src/FacePosition.o \
./src/FacesList.o \
./src/FerNNClassifier.o \
./src/FramesList.o \
./src/IntegralImage.o \
./src/LKTracker.o \
./src/PositionList.o \
./src/SingleFrame.o \
./src/TLD.o \
./src/ValidateDetection.o \
./src/run_tld.o \
./src/tld_utils.o 

CPP_DEPS += \
./src/BBoxAtFrame.d \
./src/CalcHistPositive.d \
./src/DataSharing.d \
./src/DetectionToRecognition.d \
./src/FacePosition.d \
./src/FacesList.d \
./src/FerNNClassifier.d \
./src/FramesList.d \
./src/IntegralImage.d \
./src/LKTracker.d \
./src/PositionList.d \
./src/SingleFrame.d \
./src/TLD.d \
./src/ValidateDetection.d \
./src/run_tld.d \
./src/tld_utils.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/opencv -I/usr/local/include -I/usr/local/include/opencv2 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


