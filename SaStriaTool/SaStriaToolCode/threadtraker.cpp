#include "threadtraker.h"

ThreadTraker::ThreadTraker(QObject *parent, QString fileName,QString mod) : QThread(parent)
{

    fileNameVideo=fileName;
    module=mod;

}
void ThreadTraker::run(){

    QFileInfo fileInfo(fileNameVideo);
    QDir::setCurrent(fileInfo.absolutePath());
    QDir current= QDir::current();
    QString nomeProcesso;

   // current.cdUp();
    nomeProcesso= fileInfo.absolutePath() + "/Script/" + module + " " + fileInfo.fileName();
  //    nomeProcesso= "./Script/" + module + " " + fileInfo.fileName();
    qDebug() << "current Dir: " << current.absolutePath();
    qDebug() << "processo: " << nomeProcesso;
    qDebug() << "filenameVideo: " << fileInfo.absolutePath();

   //emit mess(nomeProcesso);


    QProcess process;
  //  connect(&process,SIGNAL(finished(int)),this,SLOT(on_videoDataReady(fileName)));
    //connect(&server, SIGNAL(newConnection()),this, SLOT(acceptConnectionTraker()));
    //server.listen(QHostAddress::Any, 8888);
    process.execute(nomeProcesso);



    qDebug() << "THEDRUNNUNG: " ;



}
