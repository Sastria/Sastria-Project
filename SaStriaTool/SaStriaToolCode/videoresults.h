#ifndef VIDEORESULTS_H
#define VIDEORESULTS_H

#include <QWidget>
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

namespace Ui {
class VideoResults;
}

class VideoResults : public QWidget
{
    Q_OBJECT
    
public:
    explicit VideoResults(QWidget *parent = 0);
    ~VideoResults();
    void loadFilevideo(QString videoFile,QString xmlFile);
    
private slots:
    void on_playButton_clicked();
    void on_player_positionChange(qint64 position);
     void on_slider_released();
     void on_player_durationChange(qint64 time);

protected:
     void resizeEvent(QResizeEvent* myEvent);
     bool eventFilter(QObject *obj, QEvent *event);

private:
    Ui::VideoResults *ui;
    QSlider *positionSlider;
    QMediaPlayer *mediaPlayer;
    QVideoWidget *videoWidget;
    QString currentXml;
    QMap<QString,QList<int> > person;
    QMap<QString,QList<QList<int> > > tagColor;
    int frames;
    double frameSliceTime;
    bool canResize;

    void drawPersonBar();
    void loadXmlDocument(QString filename);
    void parseFace(QXmlStreamReader& xml);
};

#endif // VIDEORESULTS_H
