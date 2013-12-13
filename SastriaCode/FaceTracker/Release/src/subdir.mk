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
../src/FeaturesTrack.cpp \
../src/FramesList.cpp \
../src/IntegralImage.cpp \
../src/PositionList.cpp \
../src/SingleFrame.cpp \
../src/TrackerKanade.cpp \
../src/ValidateDetection.cpp \
../src/frameWk.cpp \
../src/myClassifier.cpp \
../src/run_main.cpp 

OBJS += \
./src/BBoxAtFrame.o \
./src/CalcHistPositive.o \
./src/DataSharing.o \
./src/DetectionToRecognition.o \
./src/FacePosition.o \
./src/FacesList.o \
./src/FeaturesTrack.o \
./src/FramesList.o \
./src/IntegralImage.o \
./src/PositionList.o \
./src/SingleFrame.o \
./src/TrackerKanade.o \
./src/ValidateDetection.o \
./src/frameWk.o \
./src/myClassifier.o \
./src/run_main.o 

CPP_DEPS += \
./src/BBoxAtFrame.d \
./src/CalcHistPositive.d \
./src/DataSharing.d \
./src/DetectionToRecognition.d \
./src/FacePosition.d \
./src/FacesList.d \
./src/FeaturesTrack.d \
./src/FramesList.d \
./src/IntegralImage.d \
./src/PositionList.d \
./src/SingleFrame.d \
./src/TrackerKanade.d \
./src/ValidateDetection.d \
./src/frameWk.d \
./src/myClassifier.d \
./src/run_main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -I/usr/local/include/opencv -I/usr/local/include -I/usr/local/include/opencv2 -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


