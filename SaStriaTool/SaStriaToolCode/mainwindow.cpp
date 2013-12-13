#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    ui->loadButto->setIcon(ui->loadButto->style()->standardIcon(QStyle::SP_DialogOpenButton));

    ui->panelCurrentProcess->setMinimumHeight(0);
    ui->panelCurrentProcess->setMaximumHeight(0);
    ui->progressBarDetection->setStyle(QStyleFactory::create("fusion"));
    ui->progressBarDetection->setValue(0);
    ui->progressBarTracker->setStyle(QStyleFactory::create("fusion"));
    ui->progressBarTracker->setValue(0);
    ui->progressBarRecognition->setStyle(QStyleFactory::create("fusion"));
    ui->progressBarRecognition->setValue(0);
    ui->centralWidget->setLayout(ui->wIPLayout);

}

MainWindow::~MainWindow()
{
    delete ui;
}
void MainWindow::startReadRecognition(){
    //char buffer[1024] = {0};
    QString buffer;

    //do{
       buffer= client->read(client->bytesAvailable());
       ui->progressBarRecognition->setValue(buffer.toInt());
      // qDebug() << "IL MIO VALORE LETTO IL MIO VALORE LETTO IL MIO VALORE LETTO IL MIO VALORE LETTO IL MIO VALORE LETTO" << buffer;
    //}while(buffer.compare("-1")==0);

    //client->close();
}



void MainWindow::startReadTraker(){
    //char buffer[1024] = {0};
    QString buffer;

    //do{
       buffer= client->read(client->bytesAvailable());
       ui->progressBarTracker->setValue(buffer.toInt());
}

void MainWindow::startReadDetection(){
    //char buffer[1024] = {0};
    QString buffer;


       buffer= client->read(client->bytesAvailable());
       ui->progressBarDetection->setValue(buffer.toInt()+1);

}

void MainWindow::acceptConnectionDetection(){
   // QMessageBox::information(this,"Connenction Detection","Connenction Detection",QMessageBox::Ok);

    client = server.nextPendingConnection();
    connect(client, SIGNAL(readyRead()),this, SLOT(startReadDetection()));
}

void MainWindow::acceptConnectionTracker(){
   // QMessageBox::information(this,"Connenction Traker","Connenction Traker",QMessageBox::Ok);

   // client->close();
    client = server.nextPendingConnection();
    connect(client, SIGNAL(readyRead()),this, SLOT(startReadTraker()));
}


void MainWindow::acceptConnectionRecognition(){
  // QMessageBox::information(this,"Connenction recognition","Connenction recognition",QMessageBox::Ok);

   // client->close();
    client = server.nextPendingConnection();
    connect(client, SIGNAL(readyRead()),this, SLOT(startReadRecognition()));
}


void MainWindow::startDetection(){

    ui->listProcessed->addItem(ui->labelCurrentJob->text());

    if(ui->listToProess->count()>0){
       // ui->panelCurrentProcess->setMaximumHeight(100);
       // ui->panelCurrentProcess->setMinimumHeight(100);
        ui->progressBarDetection->setValue(0);
        ui->progressBarTracker->setValue(0);
        ui->progressBarRecognition->setValue(0);

        ui->labelCurrentJob->setText(ui->listToProess->takeItem(0)->text());

       // if(server.isListening()){
        client->close();
            server.close();
            disconnect(&server, SIGNAL(newConnection()),this, SLOT(acceptConnectionRecognition()));
        //}
        connect(&server, SIGNAL(newConnection()),this, SLOT(acceptConnectionDetection()));
        server.listen(QHostAddress::Any, 8887);

        delete(TrackerT);
        TrackerT=new ThreadTraker(this,ui->labelCurrentJob->text(),"FaceDetection");
        connect(TrackerT,SIGNAL(finished()),this,SLOT(startTraker()));
        TrackerT->start();
    }else{
        ui->panelCurrentProcess->setMinimumHeight(0);
        ui->panelCurrentProcess->setMaximumHeight(0);
    }
}

void MainWindow::startTraker(){
   //  QMessageBox::information(this,"Start Traker","Start Traker",QMessageBox::Ok);

    client->close();
    server.close();
    disconnect(&server, SIGNAL(newConnection()),this, SLOT(acceptConnectionDetection()));
    connect(&server, SIGNAL(newConnection()),this, SLOT(acceptConnectionTracker()));
    server.listen(QHostAddress::Any, 8888);

    delete(TrackerT);
    TrackerT=new ThreadTraker(this,ui->labelCurrentJob->text(),"Tracker");
    connect(TrackerT,SIGNAL(finished()),this,SLOT(startRecognition()));
    TrackerT->start();
}

void MainWindow::onTrakerMess(QString str){
    QMessageBox::information(this,"Start Recognition",str,QMessageBox::Ok);
}

void MainWindow::startRecognition(){
  //QMessageBox::information(this,"Start Recognition","Start Recognition",QMessageBox::Ok);

    // create and listen for soket connection From Traker Data
    client->close();
    server.close();
    disconnect(&server, SIGNAL(newConnection()),this, SLOT(acceptConnectionTracker()));
    connect(&server, SIGNAL(newConnection()),this, SLOT(acceptConnectionRecognition()));
    server.listen(QHostAddress::Any, 8889);

    delete(TrackerT);
    TrackerT=new ThreadTraker(this,ui->labelCurrentJob->text(),"Recognition.sh");
    connect(TrackerT,SIGNAL(finished()),this,SLOT(startDetection()));
    //   connect(TrackerT,SIGNAL(mess(QString)),this,SLOT(onTrakerMess(QString)));
    TrackerT->start();
}

void MainWindow::on_loadButto_clicked()
{
    QDir current;
    QString currentXmlFile;
    QString tmpWorkString;
    ui->progressBarDetection->setValue(0);
    ui->progressBarTracker->setValue(0);
    ui->progressBarRecognition->setValue(0);

    current=QDir::current();
    scriptPath=current.absolutePath();
    qDebug() << "SCRIPT PATH: " << scriptPath;

    fileToProcess = QFileDialog::getOpenFileNames(this, tr("Open Movie"),QDir::currentPath());

    QFileInfo fileInfo(fileToProcess.at(0));    

    QFileInfo fileInfoTraker;
    //fileInfo.absoluteDir().cdUp();
    videoPath=fileInfo.absolutePath();
    QDir::setCurrent(fileInfo.absolutePath());
    current= QDir::current();
    current.cdUp();
    //workPath=current.absolutePath();
 //   qDebug() << "Work:" << workPath << " video:" << videoPath << endl;
       qDebug() << "videoPath:" << videoPath << endl;


    QProcess process;

    if(fileInfo.exists()){
        qDebug() << "FILE VIDEO ESISTENTE";
        tmpWorkString="mkdir " + videoPath + "/Data";
        process.execute(tmpWorkString);
        //tmpWorkString="cp -R " + scriptPath + "/haar " + videoPath + "/Data/haar";
        //process.execute(tmpWorkString);
        tmpWorkString="mkdir " + videoPath + "/Script";
        process.execute(tmpWorkString);
        tmpWorkString="cp " + scriptPath +"/SaStriaTool.app/Contents/MacOS/FaceDetection " + videoPath + "/Script";
        qDebug() << "workString:" << tmpWorkString;
        process.execute(tmpWorkString);
        tmpWorkString="cp " + scriptPath + "/SaStriaTool.app/Contents/MacOS/Tracker " + videoPath + "/Script";
        process.execute(tmpWorkString);
        tmpWorkString="cp " + scriptPath + "/SaStriaTool.app/Contents/MacOS/Recognition.sh " + videoPath + "/Script";
        process.execute(tmpWorkString);
        tmpWorkString="mkdir " + videoPath + "/VideoOut";
        process.execute(tmpWorkString);
        tmpWorkString="cp -R " + scriptPath + "/SaStriaTool.app/Contents/MacOS/Parameters " + videoPath + "/Parameters";
        process.execute(tmpWorkString);
        tmpWorkString="cp -R " + scriptPath + "/SaStriaTool.app/Contents/MacOS/quoqueisaall " + videoPath + "/quoqueisaall";
        process.execute(tmpWorkString);
        tmpWorkString="cp -R " + scriptPath + "/SaStriaTool.app/Contents/MacOS/lib " + videoPath + "/lib";
        process.execute(tmpWorkString);

        QStringList::Iterator it = fileToProcess.begin();
        while(it != fileToProcess.end()) {
            fileInfo.setFile(*it);
            currentXmlFile=videoPath + "/Data/RecfaceList" + fileInfo.fileName() + ".xml";
            qDebug() << "XML FILE TO READ: " <<currentXmlFile << endl;
            // if traker in the video file was not already run then create and run Traker on video File
            //QFileInfo fileInfoTraker(xmlFile);
            fileInfoTraker.setFile(currentXmlFile);
            if(fileInfoTraker.exists()){
                ui->listProcessed->addItem(fileInfo.fileName());
            }else {
                ui->listToProess->addItem(fileInfo.fileName());
            }
            ++it;
        }


        if(ui->listToProess->count()>0){
            ui->panelCurrentProcess->setMaximumHeight(100);
            ui->panelCurrentProcess->setMinimumHeight(100);
            // ui->personLabels->setMaximumHeight(0);

            ui->labelCurrentJob->setText(ui->listToProess->takeItem(0)->text());

            if(server.isListening())
                disconnect(&server, SIGNAL(newConnection()),this, SLOT(acceptConnectionRecognition()));
            connect(&server, SIGNAL(newConnection()),this, SLOT(acceptConnectionDetection()));
            server.listen(QHostAddress::Any, 8887);

            TrackerT=new ThreadTraker(this,ui->labelCurrentJob->text(),"FaceDetection");
            connect(TrackerT,SIGNAL(finished()),this,SLOT(startTraker()));
            TrackerT->start();
        }
    }else qDebug() << "FILE VIDEO NON ESISTENTE";
}

void MainWindow::on_listProcessed_itemDoubleClicked(QListWidgetItem *item)
{

    videoResult.loadFilevideo(videoPath+"/" + item->text(),videoPath + "/Data/RecfaceList" + item->text() + ".xml");
    videoResult.show();
}

void MainWindow::on_listProcessed_doubleClicked(const QModelIndex &index)
{

}
