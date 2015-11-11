# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:27:34 2014

@author: s1105
"""
"""
Liest entweder ein Stereo- oder Audio-Signal ein, berechnet daraus mit Hilfe 
der FFT die Geschwindigkeit. Diese wird in Form der aktuellen Geschwindigkeit 
und als Plot der letzten 20 s auf einem Bildschirm ausgegeben. Der parallele
Ablauf von Einlesen, Verarbeitung und Ausgeben wird mit Threads realisiert.
"""
import matplotlib
matplotlib.use('tkagg')
import alsaaudio
import queue
import threading
import numpy as np
import tkinter as tk

#fuer ausgabe
from matplotlib.backends.backend_tkagg import \
    FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

#cfar
from cfar1 import finder
#from auspunktemax import explfinder2


class Parameter():
    def __init__(self):
        #Umrechungsfaktor Geschwindigkeit in Dopplerfrequenz
        self.geschw2DopplerFreq = 44.7
        #Laenge Puffer in Soundkarte bei Aufnahme
        self.N=2048
        #Abtastfrequenz der Soundkarte
        self.fs = 44100
        #mono oder stereo
        self.kanaele = 1
        
        #neue Werte fuer Berchnungen (da zwei mal hintereinander eingelesen wird, 
        #dabei aber nur jeder 4. Werte verwendet wird)
        self.Nneu = self.N/2
        self.fsneu = self.Nneu*self.fs/(self.N*2)
        #FFT Laenge (Zero Padding)
        self.Nfft = 2048*2
        
        #cfar = True bedeutet cfar wird benutzt, sonst nur max
        self.cfar = False
        self.L = 80
        self.k = 60
        self.alpha = 2.1
        #sinnvoller Bereich der FFT anpassen
        self.strecken = 0.4
        self.messbareF = 1500    
        #Laenge Zeitplot
        self.Lzeitplt = 200
        #Faktor um Grenze fuer FFT zu berechnen
        self.grenzeFaktor = 5
        
        #Anzahl Mittelungen fuers Rauschen
        self.M = 6
        
class leseThread (threading.Thread):    
    """
    Liest das Audiosignal zwei mal hintereinadner ein, verwendet dabei 
    nur jedes 4. Element und schreibt sie in die dataQueue. Zu Beginn wird paar 
    Mal Rauschen eingelesen und in die rauschenQueue geschrieben. 
    
    dataQueue: eigentliches Zeitsignal
    rauschenQueue: Zeitsignal des Rauschens
    """
    def __init__(self, dataQueue, rauschenQueue, par):
        threading.Thread.__init__(self)
        self.par = par
        #audio-parameter
        self.inp = (alsaaudio.PCM
            (alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL, 'hw:0,0'))
        self.inp.setchannels(self.par.kanaele)
        if self.par.kanaele != self.inp.setchannels(self.par.kanaele):
            print('Kanaele konnte nicht eingestellt werden')           
        self.inp.setrate(self.par.fs)
        if self.par.fs != self.inp.setrate(self.par.fs):
            print('fs konnte nicht eingestellt werden')
        self.inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        if (alsaaudio.PCM_FORMAT_S16_LE != 
                self.inp.setformat(alsaaudio.PCM_FORMAT_S16_LE)):
            print('Datentyp konnte nicht eingestellt werden')
        self.inp.setperiodsize(self.par.N)
        if self.par.N != self.inp.setperiodsize(self.par.N):
            print('N konnte nicht eingestellt werden')
        #queues
        self.dataQueue = dataQueue
        self.rauschenQueue = rauschenQueue

    def einfachesEinlesen(self):
        #einlesen
        l, data = self.inp.read()
        #wenn nicht korrekt eingelesen wurde, nochmal einlesen
        fehler = 0
        while len(data) != 2*self.par.N:
            l, data = self.inp.read()
            fehler += 1
            #wenn zu oft falsch eingelesen wurde
            if fehler > 3:
                print('es wird die falsche Laenge eingelesen')
                print(len(data), 'len(data)')
        #datentyp richtig interpretieren
        gewandelt = np.frombuffer(data, dtype=np.dtype('<i2'))
        #nur jedes 4. Element verwenden
        if self.par.kanaele == 1:
            data4 = gewandelt[0:self.par.N-1:4]
        if self.par.kanaele == 2:
            n = 0
            data4 = np.empty(0, dtype=np.dtype('<i2'))
            while n < (len(gewandelt)):
                data4 = np.append(data4, gewandelt[n])
                data4 = np.append(data4, gewandelt[n+1])
                n += 8
        return data4            
    
    def einlesen(self):
        data = self.einfachesEinlesen()
        #nochmal einlesen
        data = np.append(data, self.einfachesEinlesen())
        return data            
        
    def referenzieren(self):
        #evtl noch alte Daten im Puffer --> vorher leer raeumen
        self.einlesen()
        self.einlesen()
        #rauschen einlesen
        for m in range(self.par.M):
            rauschen = self.einlesen()
            #in queue schreiben
            self.rauschenQueue.put(rauschen)
            
    def run(self):
        self.referenzieren()
        #evtl noch alte Daten im Puffer --> vorher leer raeumen
        self.einlesen()
        self.einlesen()
        while True:
            data = self.einlesen()
            #in queue schreiben
            self.dataQueue.put(data)
        
class fftThread (threading.Thread):
    """
    Berechnet vom Rauschen die FFTs und mittelt diese. Nach jeder eigentlichen
    Messung wird das Rauschen nach der FFT-Berechnung abgezogen. Aus Median
    des Rauschens wird Grenze gebildet. Liegen FFT-Werte oberhalb, handelt es
    sich um eine BEwegung, wenn die unterhalb liegen, dann nicht. 
    Alternativ kann CFAR verwendet werden.
    
    dataQueue: eigentliches Zeitsignal
    rauschenQueue: Zeitsignal des Rauschens
    plotQueue: berechnete Geschwindigkeiten
    """
    def __init__(self, dataQueue, plotQueue, rauschenQueue, par):
        threading.Thread.__init__(self)
        self.par = par
        self.dataQueue = dataQueue
        self.plotQueue = plotQueue
        self.rauschenQueue = rauschenQueue
        #Grenze fuer Frequenzbereich der FFT
        self.sinnvolleUntereGrenze = int(self.par.Nfft/2 
        -(self.par.Nfft/self.par.fsneu*self.par.messbareF + 
        self.par.L*self.par.strecken))
        self.sinnvolleObereGrenze = int(self.par.Nfft/2 
        + self.par.Nfft/self.par.fsneu*self.par.messbareF + 
        self.par.L*self.par.strecken)

    def referenzieren(self):
        mittelRauschenAbsFft = np.zeros(self.par.Nfft)
        mittelRauschenAbsFft = (mittelRauschenAbsFft
            [self.sinnvolleUntereGrenze : self.sinnvolleObereGrenze]) 
        for t in range(self.par.M):
            #Daten von queue holen
            rauschen = self.rauschenQueue.get()
            if self.par.kanaele == 2:
                real = data2[0:2*N-1:2]
                img = data2[1:2*N:2]
                komplex = real + 1j*img
            #Fenster
            fenster = 0.54 + 0.46 * np.cos(2 * np.pi * 
            np.arange(-self.par.Nneu/2, self.par.Nneu/2) / self.par.Nneu)
            if self.par.kanaele == 2:
                rauschenGefenstert = komplex * fenster
            else:
                rauschenGefenstert = rauschen * fenster
            #fft
            rauschenfft = np.fft.fftshift(
            np.fft.fft(rauschenGefenstert, self.par.Nfft))
            absRauschenFft = np.abs(rauschenfft)
            #sinnvoller Bereich
            absRauschenFft = (absRauschenFft
                [self.sinnvolleUntereGrenze : self.sinnvolleObereGrenze]) 
            #rauschen summieren
            mittelRauschenAbsFft = mittelRauschenAbsFft + absRauschenFft
        #Mittelwert
        mittelRauschenAbsFft = mittelRauschenAbsFft/self.par.M    
        return mittelRauschenAbsFft
    
    def run(self):
        rauschen = self.referenzieren()
        while True:
            #Daten von queue holen
            data2 = self.dataQueue.get()
            if self.par.kanaele == 2:
                real = data2[0:2*self.par.N-1:2]
                img = data2[1:2*self.par.N:2]
                komplex = real + 1j*img
            #Fenster
            fenster = 0.54 + 0.46 * np.cos(2 * np.pi 
            * np.arange(-self.par.Nneu/2, self.par.Nneu/2) / self.par.Nneu)
            if self.par.kanaele == 2:
                a = komplex * fenster
            else:
                a = data2 * fenster
            #fft
            fft = np.fft.fftshift(np.fft.fft(a, self.par.Nfft))
            absfft = np.abs(fft)
            frequenz = np.fft.fftshift(np.fft.fftfreq(
            self.par.Nfft,1/self.par.fsneu))
            #sinnvoller Bereich
            geschw = (frequenz
                [self.sinnvolleUntereGrenze : self.sinnvolleObereGrenze]/self.par.geschw2DopplerFreq)
            amplitude = (absfft
                [self.sinnvolleUntereGrenze : self.sinnvolleObereGrenze]) 
            amplitudeOhneRauschen = np.abs(amplitude-rauschen)
            #grenze
            medi = np.median(rauschen)
            grenze = self.par.grenzeFaktor*medi            
            #peaks finden  
            if self.par.cfar == True:
                punkte, schranke = finder(
                    amplitudeOhneRauschen, 
                    self.par.L, self.par.k, self.par.alpha)
                if punkte.shape[-1] != 0:
                    #nur groessten peak
                    peaks = punkte[np.argmax(amplitudeOhneRauschen[punkte])] 
                    if amplitudeOhneRauschen[peaks] > grenze:
                        #auf Frequenzachse bringen
                        peaksvonfft = geschw[peaks]
                    else:
                        peaksvonfft = 0
                else:
                    peaksvonfft = 0
                #Geschwindigkeit in Queue schreiben
                self.plotQueue.put(np.abs(peaksvonfft))                    
            else:
                #Frequenz, an der FFT maximal
                peaks = np.argmax(amplitudeOhneRauschen)
                #wenn Wert oberhalb Grenze, dann ist Bewegung vorhanden
                if amplitudeOhneRauschen[peaks] > grenze:
                    peaksvonfft = geschw[peaks]
                else:
                    peaksvonfft = 0
                #Geschwindigkeit in Queue schreiben
                self.plotQueue.put(np.abs(peaksvonfft))   
                
class ausgabe (object):
    """
    Nimmt die Geschwindigkeiten gesammelt aus der Queue. Gibt den aktuellsten 
    Wert auf dem Bildschirm aus und plottet die Geschwindigkeiten der letzten
    20s.    
    
    plotQueue: berechnete Geschwindigkeiten
    """
    #Zeit nach der neu geplottet wird
    sleepTime = 20
    #Geometrie Fenster
    XPOS = 0
    YPOS = 0
    WIDTH = 400
    HEIGHT = 400
    def __init__(self, plotQueue, par):
        self.par = par
        self.plotQueue = plotQueue
        self.zeitplt = np.zeros(self.par.Lzeitplt)
        
        #Plot
        self.fig = Figure(figsize=(12,6), dpi=100)    
        axis = (self.fig.add_subplot
            (111,title=
            'Entwicklung der Geschwindigkeit in den letzten Sekunden', 
            xlabel='Zeit in s', ylabel='Betrag Geschwindigkeit in km/h'))
        axis.set_ylim([0, 35])
        #Schriftgroesse Plot
        matplotlib.rcParams.update({'font.size': 26})
        #x-Achse in Sekunden (Umrechnung aus N und fs)
        self.line, = (axis.plot
            (np.arange(-(self.par.Lzeitplt - 1), 1)*2*self.par.N/self.par.fs,
                 np.zeros(self.par.Lzeitplt)))
    
        self.root = tk.Tk()   
        
        #Vollbild
        self.WIDTH = self.root.winfo_screenwidth() 
        self.HEIGHT = self.root.winfo_screenheight() 
        self.root.geometry('{0}x{1}+{2}+{3}'.format
        (self.WIDTH, self.HEIGHT, self.XPOS, self.YPOS))
        
        #Textfenster
        textFenster = (tk.Label
            (self.root, text = 'Ihre aktuelle Geschwindigkeit in km/h:'))
        textFenster.config(height=1, width=40)
        textFenster.config(font=('times', 40))
        textFenster.pack()

        #Geschwindigkeitsfenster        
        self.geschwFenster = tk.Label(self.root, text = 'start')
        self.geschwFenster.config(height=1, width=40)
        self.geschwFenster.config(font=('times', 70), fg="red")
        self.geschwFenster.pack()

        #canvas        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        #Aufruf von Update nach sleepTime ms     
        self.root.after(self.sleepTime, self.update)
    
    def update(self):
        """
        erneuert die Geschwindigkeitswerte und ruft sich selbst wieder neu auf     
        """
        puffer = []
        #plotQueue komlett auslesen
        try:
            while True:
                puffer.append(self.plotQueue.get_nowait())
        #Queue vollstaendig ausgelesen
        except queue.Empty:
            pass
        
        if len(puffer) > 0:
            #geschwindigkeit ausgeben
            self.geschwFenster.config(text='{:0.2f}'.format(puffer[-1]))
            self.geschwFenster.pack()
            #Werte zurueckschieben
            self.zeitplt[:-len(puffer)] = self.zeitplt[len(puffer):]
            #puffer anhaengen
            self.zeitplt[self.par.Lzeitplt-len(puffer):self.par.Lzeitplt] = puffer
            #update y-Data
            self.line.set_ydata(self.zeitplt) 
            self.canvas.draw()
        #sich selbst wieder aufrufen    
        self.root.after(self.sleepTime, self.update)


def main():
    #Main wird ausgeführt, wenn das Programm vom Terminal aus gestartet wird
    par = Parameter()
    
    dataQueue = queue.Queue()
    rauschenQueue = queue.Queue()
    plotQueue = queue.Queue()
   
    #Aufruf Ausgabe
    threadP = ausgabe(plotQueue, par) 
    
    threads = []
  
    #Start Lese-Thread
    threadL = leseThread(dataQueue, rauschenQueue, par)
    threadL.start()
    threads.append(threadL)
    
    #Start FFT-Thread
    threadF = fftThread(dataQueue, plotQueue, rauschenQueue, par)
    threadF.start()
    threads.append(threadF)
    
    tk.mainloop()  
    
    
    #warten, bis alle threads fertig
    for t in threads:
        t.join()

if __name__ == '__main__':
    main()
