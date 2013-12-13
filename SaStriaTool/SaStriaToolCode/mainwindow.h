#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QFileDialog>
#include <QMediaPlayer>
#include <QSlider>
#include <QVideoWidget>
#include <QString>
#include <QXmlStreamReader>
#include <QMessageBox>
#include <QPainter>
#include <QPixmap>
#include <QRect>
#include <QMapIterator>
#include <QResizeEvent>
#include <QColor>
#include <QProcess>
#include "QTcpServer"
#include "QTcpSocket"
#include <threadtraker.h>
#include <QStyleFactory>
#include <QListWidgetItem>
#include <videoresults.h>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT
    
public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();


    
private slots:
    void on_loadButto_clicked();
    void acceptConnectionDetection();
    void acceptConnectionRecognition();
    void acceptConnectionTracker();
    void startReadRecognition();
    void startReadTraker();
    void startReadDetection();
    void startRecognition();
    void startTraker();
    void startDetection();
    void onTrakerMess(QString str);

    void on_listProcessed_itemDoubleClicked(QListWidgetItem *item);

    void on_listProcessed_doubleClicked(const QModelIndex &index);

private:
    Ui::MainWindow *ui;
    QStringList fileToProcess;
    QString videoPath;
   // QString workPath;
    QString scriptPath;
    VideoResults videoResult;


    //Client e Server for socket Comunication
    QTcpServer server;
    QTcpSocket *client;

    //thread to execute video elaboration
    ThreadTraker *TrackerT;


};

#endif // MAINWINDOW_H
