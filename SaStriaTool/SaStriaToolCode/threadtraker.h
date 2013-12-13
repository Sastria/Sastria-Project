#ifndef THREADTRAKER_H
#define THREADTRAKER_H

#include <QString>
#include <QFileInfo>
#include <QThread>
#include <QDir>
#include <QDebug>
#include <QMessageBox>
#include <QProcess>
#include <QObject>

class ThreadTraker : public QThread
{
    Q_OBJECT
protected:
    void run();
private:
    QString nomeProcesso;
    QString fileNameVideo;
    QString module;
    QObject *par;
public:
    explicit ThreadTraker(QObject *parent = 0,QString fileName="",QString mod="");
    
signals:
   // void mess(QString str);
    
public slots:
    
};

#endif // THREADTRAKER_H
