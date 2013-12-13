install_name_tool -id lib/libavformat.54.dylib libavformat.54.dylib 
install_name_tool -id lib/libfaac.0.dylib libfaac.0.dylib
install_name_tool -id lib/libswscale.2.dylib libswscale.2.dylib 
install_name_tool -id lib/libavcodec.54.dylib libavcodec.54.dylib 
install_name_tool -id lib/libavutil.51.dylib libavutil.51.dylib
install_name_tool -id lib/libmp3lame.0.dylib libmp3lame.0.dylib
 


install_name_tool -change @executable_path/../lib/libavcodec.54.dylib lib/libavcodec.54.dylib  libavformat.54.dylib  
install_name_tool -change @executable_path/../lib/libavutil.51.dylib lib/libavutil.51.dylib  libavformat.54.dylib
install_name_tool -change @executable_path/../lib/libfaac.0.dylib lib/libfaac.0.dylib libavformat.54.dylib 
install_name_tool -change @executable_path/../lib/libmp3lame.0.dylib lib/libmp3lame.0.dylib libavformat.54.dylib 



install_name_tool -change @executable_path/../lib/libavutil.51.dylib lib/libavutil.51.dylib  libswscale.2.dylib 
install_name_tool -change @executable_path/../lib/libfaac.0.dylib lib/libfaac.0.dylib libswscale.2.dylib  
install_name_tool -change @executable_path/../lib/libmp3lame.0.dylib lib/libmp3lame.0.dylib libswscale.2.dylib 

install_name_tool -change @executable_path/../lib/libavutil.51.dylib lib/libavutil.51.dylib  libavcodec.54.dylib 
install_name_tool -change @executable_path/../lib/libfaac.0.dylib lib/libfaac.0.dylib libavcodec.54.dylib 
install_name_tool -change @executable_path/../lib/libmp3lame.0.dylib lib/libmp3lame.0.dylib libavcodec.54.dylib 

install_name_tool -change @executable_path/../lib/libfaac.0.dylib lib/libfaac.0.dylib libavutil.51.dylib
install_name_tool -change @executable_path/../lib/libmp3lame.0.dylib lib/libmp3lame.0.dylib libavutil.51.dylib 





