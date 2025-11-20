#!/bin/sh
 
for dir in XXXXXX/blizzard_2016/submissions/*; do 
d=`echo $dir | sed 's/XXXXXX/blizzard\/blizzard_2016\/submissions\///'`;
s='submission_directory/2016/audiobook/wav' ; 
mkdir $d;
echo $d; 

sox $dir/$s/AroundTheWorldIn80Days_00001_00019.wav silence44.wav $dir/$s/AroundTheWorldIn80Days_00001_00020.wav silence44.wav $dir/$s/AroundTheWorldIn80Days_00001_00021.wav silence44.wav $dir/$s/AroundTheWorldIn80Days_00001_00022.wav $d/AroundTheWorldIn80Days_0001.wav ; 
sox $dir/$s/AroundTheWorldIn80Days_00001_00019.wav silence48.wav $dir/$s/AroundTheWorldIn80Days_00001_00020.wav silence48.wav $dir/$s/AroundTheWorldIn80Days_00001_00021.wav silence48.wav $dir/$s/AroundTheWorldIn80Days_00001_00022.wav $d/AroundTheWorldIn80Days_0001.wav ; 
sox $dir/$s/AroundTheWorldIn80Days_00001_00019.wav silence16.wav $dir/$s/AroundTheWorldIn80Days_00001_00020.wav silence16.wav $dir/$s/AroundTheWorldIn80Days_00001_00021.wav silence16.wav $dir/$s/AroundTheWorldIn80Days_00001_00022.wav $d/AroundTheWorldIn80Days_0001.wav ; 


sox $dir/$s/AroundTheWorldIn80Days_00002_00033.wav silence44.wav $dir/$s/AroundTheWorldIn80Days_00002_00034.wav  $d/AroundTheWorldIn80Days_0002.wav ; 
sox $dir/$s/AroundTheWorldIn80Days_00002_00033.wav silence48.wav $dir/$s/AroundTheWorldIn80Days_00002_00034.wav  $d/AroundTheWorldIn80Days_0002.wav ; 
sox $dir/$s/AroundTheWorldIn80Days_00002_00033.wav silence16.wav $dir/$s/AroundTheWorldIn80Days_00002_00034.wav  $d/AroundTheWorldIn80Days_0002.wav ; 

sox $dir/$s/AroundTheWorldIn80Days_00003_00046.wav silence44.wav $dir/$s/AroundTheWorldIn80Days_00003_00047.wav  $d/AroundTheWorldIn80Days_0003.wav ;
sox $dir/$s/AroundTheWorldIn80Days_00003_00046.wav silence48.wav $dir/$s/AroundTheWorldIn80Days_00003_00047.wav  $d/AroundTheWorldIn80Days_0003.wav ;
sox $dir/$s/AroundTheWorldIn80Days_00003_00046.wav silence16.wav $dir/$s/AroundTheWorldIn80Days_00003_00047.wav  $d/AroundTheWorldIn80Days_0003.wav ;


sox $dir/$s/AroundTheWorldIn80Days_00004_00058.wav silence44.wav $dir/$s/AroundTheWorldIn80Days_00004_00059.wav  $d/AroundTheWorldIn80Days_0004.wav ;
sox $dir/$s/AroundTheWorldIn80Days_00004_00058.wav silence48.wav $dir/$s/AroundTheWorldIn80Days_00004_00059.wav  $d/AroundTheWorldIn80Days_0004.wav ;
sox $dir/$s/AroundTheWorldIn80Days_00004_00058.wav silence16.wav $dir/$s/AroundTheWorldIn80Days_00004_00059.wav  $d/AroundTheWorldIn80Days_0004.wav ;

sox $dir/$s/AroundTheWorldIn80Days_00004_00079.wav silence44.wav $dir/$s/AroundTheWorldIn80Days_00004_00080.wav  $d/AroundTheWorldIn80Days_0005.wav ;
sox $dir/$s/AroundTheWorldIn80Days_00004_00079.wav silence48.wav $dir/$s/AroundTheWorldIn80Days_00004_00080.wav  $d/AroundTheWorldIn80Days_0005.wav ;
sox $dir/$s/AroundTheWorldIn80Days_00004_00079.wav silence16.wav $dir/$s/AroundTheWorldIn80Days_00004_00080.wav  $d/AroundTheWorldIn80Days_0005.wav ;

sox $dir/$s/AroundTheWorldIn80Days_00005_00089.wav silence44.wav $dir/$s/AroundTheWorldIn80Days_00005_00090.wav silence44.wav $dir/$s/AroundTheWorldIn80Days_00005_00091.wav  silence44.wav $dir/$s/AroundTheWorldIn80Days_00005_00092.wav  silence44.wav $dir/$s/AroundTheWorldIn80Days_00005_00093.wav $d/AroundTheWorldIn80Days_0006.wav ;
sox $dir/$s/AroundTheWorldIn80Days_00005_00089.wav silence48.wav $dir/$s/AroundTheWorldIn80Days_00005_00090.wav silence48.wav $dir/$s/AroundTheWorldIn80Days_00005_00091.wav  silence48.wav $dir/$s/AroundTheWorldIn80Days_00005_00092.wav  silence48.wav $dir/$s/AroundTheWorldIn80Days_00005_00093.wav $d/AroundTheWorldIn80Days_0006.wav ;
sox $dir/$s/AroundTheWorldIn80Days_00005_00089.wav silence16.wav $dir/$s/AroundTheWorldIn80Days_00005_00090.wav silence16.wav $dir/$s/AroundTheWorldIn80Days_00005_00091.wav  silence16.wav $dir/$s/AroundTheWorldIn80Days_00005_00092.wav  silence16.wav $dir/$s/AroundTheWorldIn80Days_00005_00093.wav $d/AroundTheWorldIn80Days_0006.wav ;

sox $dir/$s/AroundTheWorldIn80Days_00005_00098.wav silence44.wav $dir/$s/AroundTheWorldIn80Days_00005_00099.wav silence44.wav $dir/$s/AroundTheWorldIn80Days_00005_00100.wav $d/AroundTheWorldIn80Days_0007.wav ;
sox $dir/$s/AroundTheWorldIn80Days_00005_00098.wav silence48.wav $dir/$s/AroundTheWorldIn80Days_00005_00099.wav silence48.wav $dir/$s/AroundTheWorldIn80Days_00005_00100.wav $d/AroundTheWorldIn80Days_0007.wav ;
sox $dir/$s/AroundTheWorldIn80Days_00005_00098.wav silence16.wav $dir/$s/AroundTheWorldIn80Days_00005_00099.wav silence16.wav $dir/$s/AroundTheWorldIn80Days_00005_00100.wav $d/AroundTheWorldIn80Days_0007.wav ;

sox $dir/$s/TheFirebird_00001_00014.wav silence44.wav $dir/$s/TheFirebird_00001_00015.wav $d/TheFirebird_0001.wav ;
sox $dir/$s/TheFirebird_00001_00014.wav silence48.wav $dir/$s/TheFirebird_00001_00015.wav $d/TheFirebird_0001.wav ;
sox $dir/$s/TheFirebird_00001_00014.wav silence16.wav $dir/$s/TheFirebird_00001_00015.wav $d/TheFirebird_0001.wav ;

sox $dir/$s/TheFirebird_00001_00020.wav silence44.wav $dir/$s/TheFirebird_00001_00021.wav $d/TheFirebird_0002.wav ;
sox $dir/$s/TheFirebird_00001_00020.wav silence48.wav $dir/$s/TheFirebird_00001_00021.wav $d/TheFirebird_0002.wav ;
sox $dir/$s/TheFirebird_00001_00020.wav silence16.wav $dir/$s/TheFirebird_00001_00021.wav $d/TheFirebird_0002.wav ;

sox $dir/$s/TheFirebird_00003_00034.wav silence44.wav $dir/$s/TheFirebird_00003_00035.wav $d/TheFirebird_0003.wav ;
sox $dir/$s/TheFirebird_00003_00034.wav silence48.wav $dir/$s/TheFirebird_00003_00035.wav $d/TheFirebird_0003.wav ;
sox $dir/$s/TheFirebird_00003_00034.wav silence16.wav $dir/$s/TheFirebird_00003_00035.wav $d/TheFirebird_0003.wav ;

sox $dir/$s/TheFirebird_00004_00050.wav silence44.wav $dir/$s/TheFirebird_00004_00051.wav $d/TheFirebird_0004.wav ;
sox $dir/$s/TheFirebird_00004_00050.wav silence48.wav $dir/$s/TheFirebird_00004_00051.wav $d/TheFirebird_0004.wav ;
sox $dir/$s/TheFirebird_00004_00050.wav silence16.wav $dir/$s/TheFirebird_00004_00051.wav $d/TheFirebird_0004.wav ;

sox $dir/$s/TheFirebird_00005_00062.wav silence44.wav $dir/$s/TheFirebird_00005_00063.wav $d/TheFirebird_0005.wav ;
sox $dir/$s/TheFirebird_00005_00062.wav silence48.wav $dir/$s/TheFirebird_00005_00063.wav $d/TheFirebird_0005.wav ;
sox $dir/$s/TheFirebird_00005_00062.wav silence16.wav $dir/$s/TheFirebird_00005_00063.wav $d/TheFirebird_0005.wav ;

sox $dir/$s/TheFirebird_00006_00074.wav silence44.wav $dir/$s/TheFirebird_00006_00075.wav $d/TheFirebird_0006.wav ;
sox $dir/$s/TheFirebird_00006_00074.wav silence48.wav $dir/$s/TheFirebird_00006_00075.wav $d/TheFirebird_0006.wav ;
sox $dir/$s/TheFirebird_00006_00074.wav silence16.wav $dir/$s/TheFirebird_00006_00075.wav $d/TheFirebird_0006.wav ;

sox $dir/$s/TwelfthNight_00001_00020.wav silence44.wav $dir/$s/TwelfthNight_00001_00021.wav $d/TwelfthNight_0001.wav ;
sox $dir/$s/TwelfthNight_00001_00020.wav silence48.wav $dir/$s/TwelfthNight_00001_00021.wav $d/TwelfthNight_0001.wav ;
sox $dir/$s/TwelfthNight_00001_00020.wav silence16.wav $dir/$s/TwelfthNight_00001_00021.wav $d/TwelfthNight_0001.wav ;

sox $dir/$s/TwelfthNight_00002_00029.wav silence44.wav $dir/$s/TwelfthNight_00002_00030.wav $d/TwelfthNight_0002.wav ;
sox $dir/$s/TwelfthNight_00002_00029.wav silence48.wav $dir/$s/TwelfthNight_00002_00030.wav $d/TwelfthNight_0002.wav ;
sox $dir/$s/TwelfthNight_00002_00029.wav silence16.wav $dir/$s/TwelfthNight_00002_00030.wav $d/TwelfthNight_0002.wav ;

sox $dir/$s/TwelfthNight_00002_00033_00125.wav silence44.wav $dir/$s/TwelfthNight_00002_00033_00126.wav silence44.wav $dir/$s/TwelfthNight_00002_00033_00127.wav silence44.wav $dir/$s/TwelfthNight_00002_00033_00128.wav silence44.wav $dir/$s/TwelfthNight_00002_00033_00129.wav $d/TwelfthNight_0003.wav ;
sox $dir/$s/TwelfthNight_00002_00033_00125.wav silence48.wav $dir/$s/TwelfthNight_00002_00033_00126.wav silence48.wav $dir/$s/TwelfthNight_00002_00033_00127.wav silence48.wav $dir/$s/TwelfthNight_00002_00033_00128.wav silence48.wav $dir/$s/TwelfthNight_00002_00033_00129.wav $d/TwelfthNight_0003.wav ;
sox $dir/$s/TwelfthNight_00002_00033_00125.wav silence16.wav $dir/$s/TwelfthNight_00002_00033_00126.wav silence16.wav $dir/$s/TwelfthNight_00002_00033_00127.wav silence16.wav $dir/$s/TwelfthNight_00002_00033_00128.wav silence16.wav $dir/$s/TwelfthNight_00002_00033_00129.wav $d/TwelfthNight_0003.wav ;

sox $dir/$s/TwelfthNight_00004_00049.wav silence44.wav $dir/$s/TwelfthNight_00004_00050.wav $d/TwelfthNight_0004.wav ;
sox $dir/$s/TwelfthNight_00004_00049.wav silence48.wav $dir/$s/TwelfthNight_00004_00050.wav $d/TwelfthNight_0004.wav ;
sox $dir/$s/TwelfthNight_00004_00049.wav silence16.wav $dir/$s/TwelfthNight_00004_00050.wav $d/TwelfthNight_0004.wav ;

sox $dir/$s/TwelfthNight_00004_00053.wav silence44.wav $dir/$s/TwelfthNight_00004_00054.wav $d/TwelfthNight_0005.wav ;
sox $dir/$s/TwelfthNight_00004_00053.wav silence48.wav $dir/$s/TwelfthNight_00004_00054.wav $d/TwelfthNight_0005.wav ;
sox $dir/$s/TwelfthNight_00004_00053.wav silence16.wav $dir/$s/TwelfthNight_00004_00054.wav $d/TwelfthNight_0005.wav ;

sox $dir/$s/TwelfthNight_00006_00079.wav silence44.wav $dir/$s/TwelfthNight_00006_00080.wav $d/TwelfthNight_0006.wav ;
sox $dir/$s/TwelfthNight_00006_00079.wav silence48.wav $dir/$s/TwelfthNight_00006_00080.wav $d/TwelfthNight_0006.wav ;
sox $dir/$s/TwelfthNight_00006_00079.wav silence16.wav $dir/$s/TwelfthNight_00006_00080.wav $d/TwelfthNight_0006.wav ;

sox $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00008.wav silence44.wav $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00009.wav $d/BrerRabbitAndTheBlackberryBush_0001.wav ;
sox $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00008.wav silence48.wav $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00009.wav $d/BrerRabbitAndTheBlackberryBush_0001.wav ;
sox $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00008.wav silence16.wav $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00009.wav $d/BrerRabbitAndTheBlackberryBush_0001.wav ;

sox $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00012.wav silence44.wav $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00013.wav $d/BrerRabbitAndTheBlackberryBush_0002.wav ;
sox $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00012.wav silence48.wav $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00013.wav $d/BrerRabbitAndTheBlackberryBush_0002.wav ;
sox $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00012.wav silence16.wav $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00013.wav $d/BrerRabbitAndTheBlackberryBush_0002.wav ;

sox $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00021.wav silence44.wav $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00022.wav $d/BrerRabbitAndTheBlackberryBush_0003.wav ;
sox $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00021.wav silence48.wav $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00022.wav $d/BrerRabbitAndTheBlackberryBush_0003.wav ;
sox $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00021.wav silence16.wav $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00022.wav $d/BrerRabbitAndTheBlackberryBush_0003.wav ;

sox $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00023.wav silence44.wav $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00024.wav $d/BrerRabbitAndTheBlackberryBush_0004.wav ;
sox $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00023.wav silence48.wav $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00024.wav $d/BrerRabbitAndTheBlackberryBush_0004.wav ;
sox $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00023.wav silence16.wav $dir/$s/BrerRabbitAndTheBlackberryBush_00000_00024.wav $d/BrerRabbitAndTheBlackberryBush_0004.wav ;

sox $dir/$s/HanselAndGretel_00002_00017.wav silence44.wav $dir/$s/HanselAndGretel_00002_00018.wav $d/HanselAndGretel_0001.wav ;
sox $dir/$s/HanselAndGretel_00002_00017.wav silence48.wav $dir/$s/HanselAndGretel_00002_00018.wav $d/HanselAndGretel_0001.wav ;
sox $dir/$s/HanselAndGretel_00002_00017.wav silence16.wav $dir/$s/HanselAndGretel_00002_00018.wav $d/HanselAndGretel_0001.wav ;

sox $dir/$s/HanselAndGretel_00002_00019.wav silence44.wav $dir/$s/HanselAndGretel_00002_00020.wav $d/HanselAndGretel_0002.wav ;
sox $dir/$s/HanselAndGretel_00002_00019.wav silence48.wav $dir/$s/HanselAndGretel_00002_00020.wav $d/HanselAndGretel_0002.wav ;
sox $dir/$s/HanselAndGretel_00002_00019.wav silence16.wav $dir/$s/HanselAndGretel_00002_00020.wav $d/HanselAndGretel_0002.wav ;

sox $dir/$s/HanselAndGretel_00003_00027.wav silence44.wav $dir/$s/HanselAndGretel_00003_00028.wav $d/HanselAndGretel_0003.wav ;
sox $dir/$s/HanselAndGretel_00003_00027.wav silence48.wav $dir/$s/HanselAndGretel_00003_00028.wav $d/HanselAndGretel_0003.wav ;
sox $dir/$s/HanselAndGretel_00003_00027.wav silence16.wav $dir/$s/HanselAndGretel_00003_00028.wav $d/HanselAndGretel_0003.wav ;

sox $dir/$s/HanselAndGretel_00003_00033_00146.wav silence44.wav $dir/$s/HanselAndGretel_00003_00033_00147.wav silence44.wav $dir/$s/HanselAndGretel_00003_00033_00148.wav silence44.wav $dir/$s/HanselAndGretel_00003_00033_00149.wav $d/HanselAndGretel_0004.wav ;
sox $dir/$s/HanselAndGretel_00003_00033_00146.wav silence48.wav $dir/$s/HanselAndGretel_00003_00033_00147.wav silence48.wav $dir/$s/HanselAndGretel_00003_00033_00148.wav silence48.wav $dir/$s/HanselAndGretel_00003_00033_00149.wav $d/HanselAndGretel_0004.wav ;
sox $dir/$s/HanselAndGretel_00003_00033_00146.wav silence16.wav $dir/$s/HanselAndGretel_00003_00033_00147.wav silence16.wav $dir/$s/HanselAndGretel_00003_00033_00148.wav silence16.wav $dir/$s/HanselAndGretel_00003_00033_00149.wav $d/HanselAndGretel_0004.wav ;

sox $dir/$s/HanselAndGretel_00004_00037.wav silence44.wav $dir/$s/HanselAndGretel_00004_00038.wav $d/HanselAndGretel_0005.wav ;
sox $dir/$s/HanselAndGretel_00004_00037.wav silence48.wav $dir/$s/HanselAndGretel_00004_00038.wav $d/HanselAndGretel_0005.wav ;
sox $dir/$s/HanselAndGretel_00004_00037.wav silence16.wav $dir/$s/HanselAndGretel_00004_00038.wav $d/HanselAndGretel_0005.wav ;

sox $dir/$s/HanselAndGretel_00006_00057.wav silence44.wav $dir/$s/HanselAndGretel_00006_00058.wav $d/HanselAndGretel_0006.wav ;
sox $dir/$s/HanselAndGretel_00006_00057.wav silence48.wav $dir/$s/HanselAndGretel_00006_00058.wav $d/HanselAndGretel_0006.wav ;
sox $dir/$s/HanselAndGretel_00006_00057.wav silence16.wav $dir/$s/HanselAndGretel_00006_00058.wav $d/HanselAndGretel_0006.wav ;

sox $dir/$s/KnightsAndCastles_00000_00011.wav silence44.wav $dir/$s/KnightsAndCastles_00000_00012.wav $d/KnightsAndCastles_0001.wav ;
sox $dir/$s/KnightsAndCastles_00000_00011.wav silence48.wav $dir/$s/KnightsAndCastles_00000_00012.wav $d/KnightsAndCastles_0001.wav ;
sox $dir/$s/KnightsAndCastles_00000_00011.wav silence16.wav $dir/$s/KnightsAndCastles_00000_00012.wav $d/KnightsAndCastles_0001.wav ;

sox $dir/$s/KnightsAndCastles_00000_00015.wav silence44.wav $dir/$s/KnightsAndCastles_00000_00016.wav $d/KnightsAndCastles_0002.wav ;
sox $dir/$s/KnightsAndCastles_00000_00015.wav silence48.wav $dir/$s/KnightsAndCastles_00000_00016.wav $d/KnightsAndCastles_0002.wav ;
sox $dir/$s/KnightsAndCastles_00000_00015.wav silence16.wav $dir/$s/KnightsAndCastles_00000_00016.wav $d/KnightsAndCastles_0002.wav ;

sox $dir/$s/KnightsAndCastles_00000_00025.wav silence44.wav $dir/$s/KnightsAndCastles_00000_00026.wav $d/KnightsAndCastles_0003.wav ;
sox $dir/$s/KnightsAndCastles_00000_00025.wav silence48.wav $dir/$s/KnightsAndCastles_00000_00026.wav $d/KnightsAndCastles_0003.wav ;
sox $dir/$s/KnightsAndCastles_00000_00025.wav silence16.wav $dir/$s/KnightsAndCastles_00000_00026.wav $d/KnightsAndCastles_0003.wav ;

sox $dir/$s/KnightsAndCastles_00000_00030.wav silence44.wav $dir/$s/KnightsAndCastles_00000_00031.wav $d/KnightsAndCastles_0004.wav ;
sox $dir/$s/KnightsAndCastles_00000_00030.wav silence48.wav $dir/$s/KnightsAndCastles_00000_00031.wav $d/KnightsAndCastles_0004.wav ;
sox $dir/$s/KnightsAndCastles_00000_00030.wav silence16.wav $dir/$s/KnightsAndCastles_00000_00031.wav $d/KnightsAndCastles_0004.wav ;

sox $dir/$s/KnightsAndCastles_00000_00032.wav silence44.wav $dir/$s/KnightsAndCastles_00000_00033.wav $d/KnightsAndCastles_0005.wav ;
sox $dir/$s/KnightsAndCastles_00000_00032.wav silence48.wav $dir/$s/KnightsAndCastles_00000_00033.wav $d/KnightsAndCastles_0005.wav ;
sox $dir/$s/KnightsAndCastles_00000_00032.wav silence16.wav $dir/$s/KnightsAndCastles_00000_00033.wav $d/KnightsAndCastles_0005.wav ;

sox $dir/$s/KnightsAndCastles_00000_00038.wav silence44.wav $dir/$s/KnightsAndCastles_00000_00039.wav $d/KnightsAndCastles_0006.wav ;
sox $dir/$s/KnightsAndCastles_00000_00038.wav silence48.wav $dir/$s/KnightsAndCastles_00000_00039.wav $d/KnightsAndCastles_0006.wav ;
sox $dir/$s/KnightsAndCastles_00000_00038.wav silence16.wav $dir/$s/KnightsAndCastles_00000_00039.wav $d/KnightsAndCastles_0006.wav ;
done
