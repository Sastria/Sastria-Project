#include "videoresults.h"
#include "ui_videoresults.h"

VideoResults::VideoResults(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::VideoResults)
{
    ui->setupUi(this);

    positionSlider = new QSlider(Qt::Horizontal);
    positionSlider->setRange(0, 0);


    videoWidget= new QVideoWidget();
    mediaPlayer = new QMediaPlayer;
    mediaPlayer->setVolume(50);
    mediaPlayer->setVideoOutput(videoWidget);
    videoWidget->setMinimumHeight(200);

    connect(mediaPlayer,SIGNAL(positionChanged(qint64)),this,SLOT(on_player_positionChange(qint64)));
    connect(positionSlider,SIGNAL(sliderReleased()),this,SLOT(on_slider_released()));



   ui->personLabels->setMaximumHeight(0);


   ui->playButton->setIcon(ui->playButton->style()->standardIcon(QStyle::SP_MediaPlay));
   ui->VideoPanelLayout->addWidget(videoWidget);
   ui->VideoPanelLayout->addWidget(positionSlider);

   positionSlider->setMinimumHeight(30);
   positionSlider->setMaximumHeight(30);
   ui->playButton->setMaximumHeight(30);
   ui->playButton->setMinimumHeight(30);

   this->setLayout(ui->cWLayout);

}

VideoResults::~VideoResults()
{
    delete ui;
}

void VideoResults::resizeEvent(QResizeEvent* myEvent)
{
    //QPixmap myPix(QSize(0,0));
   // ui->personLabels->setPixmap(myPix);
    QWidget::resizeEvent(myEvent);
    if(canResize)
       drawPersonBar();
}


bool VideoResults::eventFilter(QObject *obj, QEvent *event){

     if(event->type() == QEvent::MouseButtonPress){

           QMouseEvent *mouseEvent = static_cast<QMouseEvent *>(event);
           qDebug() << "x:" << mouseEvent->x() << " y:" << mouseEvent->y() << " Dur:"<< mediaPlayer->duration();
           mediaPlayer->setPosition(((float)mouseEvent->x()/ui->personLabels->width())*mediaPlayer->duration());
      return(true);
     }else return(obj->eventFilter(obj,event));

}




void VideoResults::loadFilevideo(QString videoFile,QString xmlFile){
    //QString videoFile=videoPath+"/" + fileName;
    //currentXml=workPath + "/Data/RecfaceList" + fileName + ".xml";
    mediaPlayer->stop();
    disconnect(mediaPlayer,SIGNAL(durationChanged(qint64)),this,SLOT(on_player_durationChange(qint64)));
    currentXml=xmlFile;
    if (!videoFile.isEmpty()) {
        QFileInfo fileInfoRec(currentXml);
        if(fileInfoRec.exists()){
           // qDebug() << fileName;
            connect(mediaPlayer,SIGNAL(durationChanged(qint64)),this,SLOT(on_player_durationChange(qint64)));
            mediaPlayer->setMedia(QUrl::fromLocalFile(videoFile));
            ui->playButton->setEnabled(true);
            positionSlider->setEnabled(true);
        }
    }
}

void VideoResults::on_player_durationChange(qint64 time){
    QMessageBox::critical(this, tr("Error"), tr("duration Chamge"));
    disconnect(mediaPlayer,SIGNAL(durationChanged(qint64)),this,SLOT(on_player_durationChange(qint64)));
    loadXmlDocument(currentXml);
    positionSlider->setRange(0,time);
    //frameSliceTime=(double)positionSlider->width()/frames;
   // frameSliceTime=frameSliceTime > 1 ? frameSliceTime:1;
   // qDebug() << endl <<"framesNr:"<< frames << " Duration Video:" << mediaPlayer->duration() << " Duration Frame:" << frameSliceTime;
    drawPersonBar();
   // ui->pBarContainer->setMaximumHeight(0);
   // ui->pBarContainer->setMinimumHeight(0);
   //this->setCurrentWidget(ui->tabProcessed);

}

void VideoResults::on_player_positionChange(qint64 position){
    positionSlider->setValue(position);
}


void VideoResults::on_slider_released(){
    mediaPlayer->setPosition(positionSlider->sliderPosition());
}


void VideoResults::on_playButton_clicked()
{
    switch(mediaPlayer->state()) {
       case QMediaPlayer::PlayingState:
           mediaPlayer->pause();
            ui->playButton->setIcon(ui->playButton->style()->standardIcon(QStyle::SP_MediaPlay));
            ui->playButton->setText("Play");
           break;
       default:
           mediaPlayer->play();
            ui->playButton->setIcon(ui->playButton->style()->standardIcon(QStyle::SP_MediaPause));
            ui->playButton->setText("Pause");
           break;
       }
}


void VideoResults::drawPersonBar(){

    ui->personLabels->setMaximumHeight(person.size()*20);
    ui->personLabels->setMinimumHeight(ui->personLabels->maximumHeight()/2);
   // ui->personLabels->setMinimumHeight(person.size()*20);
    QPixmap myPix(QSize(ui->personLabels->width(),ui->personLabels->height()));
    QPainter painter(&myPix);
    myPix.fill();
    int count=0;
    QList<int> fPos;
    QList<int> tmpc;
    int barH=(ui->personLabels->height()/person.size());
    int ypos=0;
    frameSliceTime=(double)positionSlider->width()/frames;
    float widthBB=ui->personLabels->width();
    QMapIterator<QString, QList<int> > i(person);
    QMapIterator<QString, QList<QList<int> > > tcolor(tagColor);
    QColor colorPaint;
    QColor penColorGray;
    QColor penColorBlack;

    penColorGray.setRed(230);
    penColorGray.setGreen(230);
    penColorGray.setBlue(230);

    penColorBlack.setRed(0);
    penColorBlack.setGreen(0);
    penColorBlack.setBlue(0);

    barH-=2;
    widthBB=frameSliceTime>1? frameSliceTime:1;
    ypos=0;
    while (i.hasNext()) {
        i.next();
        tcolor.next();
        //qDebug() << endl <<"Persona: "<< i.key();
        tmpc=tcolor.value().at(0);
        colorPaint.setRed(tmpc.at(0));
        colorPaint.setGreen(tmpc.at(1));
        colorPaint.setBlue(tmpc.at(2));

        fPos=i.value();

       // painter.drawText(QPoint(frameSliceTime*fPos.at(0),ypos+barH),i.key());

       // qDebug() << " frames: ";
        count=0;
        while(count<fPos.size()){
          //  qDebug() << fPos.at(count)<< ",";
            painter.fillRect(QRectF(frameSliceTime*fPos.at(count),ypos,widthBB,barH),colorPaint);
            count++;
        }
         ypos+=barH+2;


         painter.setPen(penColorGray);
         painter.drawLine(QPoint(0,ypos-2),QPoint(myPix.width(),ypos-2));
         painter.setPen(penColorBlack);
    }


    i.toFront();
    ypos=0;
    while (i.hasNext()) {
        i.next();
        //painter.setPen(Qt::blue);
        fPos=i.value();
       // qDebug() << endl <<"Persona Video: "<< i.key();
        painter.drawText(QPoint(1,ypos+((barH+8)/2)),i.key());
        ypos+=barH+2;
    }

  ui->personLabels->setPixmap(myPix);
}





void VideoResults::loadXmlDocument(QString filename){

    if (!filename.isEmpty()) {
         QFile file(filename);
         if (!file.open(QIODevice::ReadOnly)) {
                QMessageBox::critical(this, tr("Error"), tr("Could not open file"));

         }else{

             /*
                ui->progressBarDetection->setValue(0);
                ui->progressBarTracker->setValue(0);
                ui->progressBarRecognition->setValue(0);
                */
                int currentFrame=0;

                 person.clear();
                 tagColor.clear();
                /* QXmlStreamReader takes any QIODevice. */
                QXmlStreamReader xml(&file);
                //QList< QMap<QString,QString> > persons;
                /* We'll parse the XML until we reach end of it.*/
                while(!xml.atEnd() &&  !xml.hasError()) {
                    /* Read next element.*/
                    QXmlStreamReader::TokenType token = xml.readNext();

                    /* If token is StartElement, we'll see if we can read it.*/
                    if(token == QXmlStreamReader::StartElement) {
                            /* If it's named persons, we'll go to the next.*/
                            if(xml.name() == "nrFrames") {
                               xml.readNext();
                               frames=xml.text().toString().toInt();
                              // qDebug() << "frames" << frames;
                            }


                            /* If it's named person, we'll dig the information from there.*/
                            if(xml.name() == "Face") {
                                parseFace(xml);
                                currentFrame++;
                                /*
                                ui->progressBarDetection->setValue(currentFrame);
                                ui->progressBarTracker->setValue(currentFrame);
                                ui->progressBarRecognition->setValue(currentFrame);
                                */
                                qApp->processEvents();
                            }
                    }
                }

                /* Error handling. */
                if(xml.hasError()) {
                    QMessageBox::critical(this,"QXSRExample::parseXML",xml.errorString(),QMessageBox::Ok);
                }

                /* Removes any device() or data from the reader
                * and resets its internal state to the initial state. */
                xml.clear();

        }
    }

    /*
    ui->progressBarDetection->setValue(100);
    ui->progressBarTracker->setValue(100);
    ui->progressBarRecognition->setValue(100);
    */
    ui->personLabels->installEventFilter(this);
    canResize=true;
}

void VideoResults::parseFace(QXmlStreamReader &xml){

    QString peopleName;
    QList<int> peopleFramePosition;
    QList<QList<int> > ccolor;
    QList<int> rgb;

    if(xml.tokenType() != QXmlStreamReader::StartElement && xml.name() == "Face") {
         return;
     }

     /* Next element... */

     while(!(xml.tokenType() == QXmlStreamReader::EndElement && xml.name() == "Face")) {


         if(xml.tokenType() == QXmlStreamReader::StartElement) {

                 if(xml.name() == "label") {
                    xml.readNext();
                   // qDebug() << endl<< endl << "NOME: " <<xml.text();
                    peopleName=xml.text().toString();
                 }

                 if(xml.name() == "r") {
                    xml.readNext();
                   // qDebug() << endl<< endl << "red: " <<xml.text();
                    rgb.append(xml.text().toString().toInt());
                 }


                 if(xml.name() == "g") {
                    xml.readNext();
                  //  qDebug() << endl<< endl << "green: " <<xml.text();
                   rgb.append(xml.text().toString().toInt());
                 }

                 if(xml.name() == "b") {
                    xml.readNext();
                  //  qDebug() << endl<< endl << "blue: " <<xml.text();
                    rgb.append(xml.text().toString().toInt());
                 }


                 /* We've found first name. */
                 if((xml.name() == "BBox") ) {
                   //  qDebug() << endl << "----BBOX" << endl;
                      while(!(xml.tokenType() == QXmlStreamReader::EndElement && xml.name() == "BBox")) {

                       if((xml.tokenType() == QXmlStreamReader::StartElement)) {
                          /* We've found frameId. */
                          if(xml.name() == "frameId") {
                              xml.readNext();
                            //  qDebug() << "      FID:" <<xml.text()<< endl;
                              peopleFramePosition.append(xml.text().toString().toInt());
                          }
                          /* We've found x. */
                          if(xml.name() == "x") {
                              xml.readNext();
                            //   qDebug() << "      x:" <<xml.text()<< endl;
                          }
                          /* We've found y. */
                          if(xml.name() == "y") {
                              xml.readNext();
                            //   qDebug() << "      y:" <<xml.text()<< endl;
                          }
                          /* We've found w. */
                          if(xml.name() == "w") {
                              xml.readNext();
                           //    qDebug() << "      w:" <<xml.text()<< endl;
                          }
                          /* We've found h. */
                          if(xml.name() == "h") {
                              xml.readNext();
                            //   qDebug() << "      h:" <<xml.text()<< endl;
                          }
                          /* We've found confidence. */
                          if(xml.name() == "confidence") {
                              xml.readNext();
                             //  qDebug() << "      cnf:" <<xml.text()<< endl;
                          }
                          /* ...and next... */
                       }
                          xml.readNext();
                      }
                 }

             }
             /* ...and next... */
             xml.readNext();

         }

     ccolor.append(rgb);
     tagColor.insert(peopleName,ccolor);
     person.insert(peopleName,peopleFramePosition);

}
