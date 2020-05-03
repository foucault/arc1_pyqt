####################################

# (c) Radu Berdan
# ArC Instruments Ltd.

# This code is licensed under GNU v3 license (see LICENSE.txt for details)

####################################

from PyQt5 import QtGui, QtCore, QtWidgets
import sys
import os
import time
import importlib

import pyqtgraph as pg
import numpy as np

import Graphics
import Globals.GlobalFonts as fonts
import Globals.GlobalFunctions as f
from Globals.modutils import BaseThreadWrapper, BaseProgPanel, makeDeviceList
import Globals.GlobalVars as g
import Globals.GlobalStyles as s
import ProgPanels

tag="CT"
g.tagDict.update({tag:"CurveTracer*"})


def _max_without_inf(lst, exclude):
    maxim = 0
    for value in lst:
        if type(value) == list:
            value = _max_without_inf(value, exclude)
            if value > maxim:
                maxim = value
        else:
            if value > maxim and value != exclude:
                maxim = value

    return maxim


def _min_without_inf(lst, exclude):
    maxim = 1e100
    for value in lst:
        if type(value) == list:
            value = _min_without_inf(value, exclude)
            if value < maxim:
                maxim = value
        else:
            if value < maxim and value != exclude:
                maxim = value

    return maxim


class ThreadWrapper(BaseThreadWrapper):

    # As CurveTracer has a programmable Vread we need to override the
    # sendData signal to use the full form of `GlobalFunctions.updateHistory`
    # which also accepts an additional argument at the end for the current
    # Vread.
    sendData = QtCore.pyqtSignal(int, int, float, float, float, str, float)

    def __init__(self, deviceList, totalCycles):
        super().__init__()
        self.deviceList=deviceList
        self.totalCycles=totalCycles

    @BaseThreadWrapper.runner
    def run(self):

        global tag

        readTag='R'+str(g.readOption)+' V='+str(g.Vread)

        g.ser.write_b(str(int(len(self.deviceList)))+"\n")

        for device in self.deviceList:
            w=device[0]
            b=device[1]
            self.highlight.emit(w,b)

            g.ser.write_b(str(int(w))+"\n")
            g.ser.write_b(str(int(b))+"\n")

            firstPoint=1
            for cycle in range(1,self.totalCycles+1):

                endCommand=0

                valuesNew=f.getFloats(3)

                if (float(valuesNew[0])!=0 or float(valuesNew[1])!=0 or float(valuesNew[2])!=0):
                    if (firstPoint==1):
                        tag_=tag+'_s'
                        firstPoint=0
                    else:
                        tag_=tag+'_i_'+str(cycle)
                else:
                    endCommand=1

                while(endCommand==0):
                    valuesOld=valuesNew
                    valuesNew=f.getFloats(3)

                    if (float(valuesNew[0])!=0 or float(valuesNew[1])!=0 or float(valuesNew[2])!=0):
                        self.sendData.emit(w,b,valuesOld[0],valuesOld[1],valuesOld[2],tag_,valuesOld[1])
                        self.displayData.emit()
                        tag_=tag+'_i_'+str(cycle)
                    else:
                        if (cycle==self.totalCycles):
                            tag_=tag+'_e'
                        else:
                            tag_=tag+'_i_'+str(cycle)
                        self.sendData.emit(w,b,valuesOld[0],valuesOld[1],valuesOld[2],tag_,valuesOld[1])
                        self.displayData.emit()
                        endCommand=1
            self.updateTree.emit(w,b)


class CurveTracer(BaseProgPanel):

    def __init__(self, short=False):
        super().__init__(title="CurveTracer",\
                description="Standard IV measurement module with "
                "current cut-off.", short=short)
        self.initUI()

    def initUI(self):

        vbox1=QtWidgets.QVBoxLayout()

        titleLabel = QtWidgets.QLabel(self.title)
        titleLabel.setFont(fonts.font1)
        descriptionLabel = QtWidgets.QLabel(self.description)
        descriptionLabel.setFont(fonts.font3)
        descriptionLabel.setWordWrap(True)

        isInt=QtGui.QIntValidator()
        isFloat=QtGui.QDoubleValidator()

        leftLabels=['Positive voltage max (V)', \
                    'Negative voltage max (V)', \
                    'Voltage step (V)', \
                    'Start Voltage (V)', \
                    'Step width (ms)']
        self.leftEdits=[]

        rightLabels=['Cycles', \
                    'Interpulse (ms)',\
                    'Positive current cut-off (uA)',\
                    'Negative current cut-off (uA)']

        self.rightEdits=[]

        leftInit=  ['1', \
                    '1', \
                    '0.05', \
                    '0.05', \
                    '50']
        rightInit= ['1', \
                    '10',\
                    '0',\
                    '0']

        # Setup the two combo boxes
        IVtypes=['Staircase', 'Pulsed']
        IVoptions=['Start towards V+', 'Start towards V-', 'Only V+', 'Only V-']

        self.combo_IVtype=QtWidgets.QComboBox()
        self.combo_IVoption=QtWidgets.QComboBox()

        self.combo_IVtype.insertItems(1,IVtypes)
        self.combo_IVoption.insertItems(1,IVoptions)

        self.combo_IVtype.currentIndexChanged.connect(self.updateIVtype)
        self.combo_IVoption.currentIndexChanged.connect(self.updateIVoption)


        # Setup the two combo boxes
        gridLayout=QtWidgets.QGridLayout()
        gridLayout.setColumnStretch(0,3)
        gridLayout.setColumnStretch(1,1)
        gridLayout.setColumnStretch(2,1)
        gridLayout.setColumnStretch(3,1)
        gridLayout.setColumnStretch(4,3)

        if self.short==False:
            gridLayout.setColumnStretch(5,1)
            gridLayout.setColumnStretch(6,1)
            gridLayout.setColumnStretch(7,2)
        #gridLayout.setSpacing(2)

        #setup a line separator
        lineLeft=QtWidgets.QFrame()
        lineLeft.setFrameShape(QtWidgets.QFrame.VLine)
        lineLeft.setFrameShadow(QtWidgets.QFrame.Raised)
        lineLeft.setLineWidth(1)
        lineRight=QtWidgets.QFrame()
        lineRight.setFrameShape(QtWidgets.QFrame.VLine)
        lineRight.setFrameShadow(QtWidgets.QFrame.Raised)
        lineRight.setLineWidth(1)

        gridLayout.addWidget(lineLeft, 0, 2, 7, 1)
        gridLayout.addWidget(lineRight, 0, 6, 7, 1)

        for i in range(len(leftLabels)):
            lineLabel=QtWidgets.QLabel()
            #lineLabel.setFixedHeight(50)
            lineLabel.setText(leftLabels[i])
            gridLayout.addWidget(lineLabel, i,0)

            lineEdit=QtWidgets.QLineEdit()
            lineEdit.setText(leftInit[i])
            lineEdit.setValidator(isFloat)
            self.leftEdits.append(lineEdit)
            gridLayout.addWidget(lineEdit, i,1)

        for i in range(len(rightLabels)):
            lineLabel=QtWidgets.QLabel()
            lineLabel.setText(rightLabels[i])
            #lineLabel.setFixedHeight(50)
            gridLayout.addWidget(lineLabel, i,4)

            lineEdit=QtWidgets.QLineEdit()
            lineEdit.setText(rightInit[i])
            lineEdit.setValidator(isFloat)
            self.rightEdits.append(lineEdit)
            gridLayout.addWidget(lineEdit, i,5)

        self.rightEdits[2].editingFinished.connect(self.imposeLimitsOnCurrentStopP)
        self.rightEdits[3].editingFinished.connect(self.imposeLimitsOnCurrentStopN)
        self.leftEdits[4].editingFinished.connect(self.imposeLimitsOnStepWidth)


        returnCheckBox = QtWidgets.QCheckBox("Halt and return.")
        returnCheckBox.stateChanged.connect(self.toggleReturn)
        self.returnCheck=0
        gridLayout.addWidget(returnCheckBox, 4, 5)

        lineLabel=QtWidgets.QLabel()
        lineLabel.setText('Bias type:')
        gridLayout.addWidget(lineLabel,5,4)

        lineLabel=QtWidgets.QLabel()
        lineLabel.setText('IV span:')
        gridLayout.addWidget(lineLabel,6,4)

        gridLayout.addWidget(self.combo_IVtype,5,5)
        gridLayout.addWidget(self.combo_IVoption,6,5)

        vbox1.addWidget(titleLabel)
        vbox1.addWidget(descriptionLabel)

        self.vW=QtWidgets.QWidget()
        self.vW.setLayout(gridLayout)
        self.vW.setContentsMargins(0,0,0,0)

        self.scrlArea=QtWidgets.QScrollArea()
        self.scrlArea.setWidget(self.vW)
        self.scrlArea.setContentsMargins(0,0,0,0)
        self.scrlArea.setWidgetResizable(False)
        self.scrlArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scrlArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)

        self.scrlArea.installEventFilter(self)

        vbox1.addWidget(self.scrlArea)
        vbox1.addStretch()

        if self.short==False:

            self.hboxProg=QtWidgets.QHBoxLayout()

            push_single = self.makeControlButton('Appy to One', \
                    self.programOne)
            push_range = self.makeControlButton('Apply to Range', \
                    self.programRange)
            push_all = self.makeControlButton('Apply to All', \
                    self.programAll)

            self.hboxProg.addWidget(push_single)
            self.hboxProg.addWidget(push_range)
            self.hboxProg.addWidget(push_all)

            vbox1.addLayout(self.hboxProg)

            push_live = QtWidgets.QPushButton("LIVE")
            push_live.setStyleSheet("background-color: red")
            push_live.clicked.connect(self.goLive)
            gridLayout.addWidget(push_live,len(self.leftEdits),0)

        self.setLayout(vbox1)
        self.vW.setFixedWidth(self.size().width())
        self.gridLayout=gridLayout

    def goLive(self):
        thisPanel = importlib.import_module(".".join([ProgPanels.__name__, "CT_LIVE"]))
        panel_class = getattr(thisPanel, "CT_LIVE")
        self.widg=panel_class(short=True)

    def imposeLimitsOnStepWidth(self):
        currentText=float(self.leftEdits[4].text())
        if (currentText<2):
            self.leftEdits[4].setText("2")

    def imposeLimitsOnCurrentStopP(self):
        currentText=float(self.rightEdits[2].text())
        if (currentText<10):
            if (currentText==0):
                self.rightEdits[2].setText("0")
            else:
                self.rightEdits[2].setText("10")

        if (currentText>1000):
            self.rightEdits[2].setText("1000")

    def imposeLimitsOnCurrentStopN(self):
        currentText=float(self.rightEdits[3].text())
        if (currentText<10):
            if (currentText==0):
                self.rightEdits[3].setText("0")
            else:
                self.rightEdits[3].setText("10")
        if (currentText>1000):
            self.rightEdits[3].setText("1000")

    def toggleReturn(self, state):
        if state == 0:
            self.returnCheck=0
        else:
            self.returnCheck=1

    def extractPanelParameters(self):
        layoutItems=[[i,self.gridLayout.itemAt(i).widget()] for i in range(self.gridLayout.count())]

        layoutWidgets=[]

        for i,item in layoutItems:
            if isinstance(item, QtWidgets.QLineEdit):
                layoutWidgets.append([i,'QLineEdit', item.text()])
            if isinstance(item, QtWidgets.QComboBox):
                layoutWidgets.append([i,'QComboBox', item.currentIndex()])
            if isinstance(item, QtWidgets.QCheckBox):
                layoutWidgets.append([i,'QCheckBox', item.checkState()])

        return layoutWidgets

    def setPanelParameters(self, layoutWidgets):
        for i,type,value in layoutWidgets:
            if type == 'QLineEdit':
                self.gridLayout.itemAt(i).widget().setText(value)
            if type == 'QComboBox':
                self.gridLayout.itemAt(i).widget().setCurrentIndex(value)
            if type == 'QCheckBox':
                self.gridLayout.itemAt(i).widget().setChecked(value)

    def updateIVtype(self, event):
        pass

    def updateIVoption(self, event):
        pass

    def eventFilter(self, object, event):
        if event.type()==QtCore.QEvent.Resize:
            self.vW.setFixedWidth(event.size().width()-object.verticalScrollBar().width())
        return False

    def resizeWidget(self,event):
        pass

    def sendParams(self):
        g.ser.write_b(str(float(self.leftEdits[0].text()))+"\n")
        g.ser.write_b(str(float(self.leftEdits[1].text()))+"\n")
        g.ser.write_b(str(float(self.leftEdits[3].text()))+"\n")
        g.ser.write_b(str(float(self.leftEdits[2].text()))+"\n")
        g.ser.write_b(str((float(self.leftEdits[4].text())-2)/1000)+"\n")
        g.ser.write_b(str(float(self.rightEdits[1].text())/1000)+"\n")
        time.sleep(0.01)
        CSp=float(self.rightEdits[2].text())
        CSn=float(self.rightEdits[3].text())

        if CSp==10.0:
            CSp=10.1
        if CSn==10.0:
            CSn=10.1


        g.ser.write_b(str(CSp/1000000)+"\n")
        g.ser.write_b(str(CSn/-1000000)+"\n")

        g.ser.write_b(str(int(self.rightEdits[0].text()))+"\n")
        g.ser.write_b(str(int(self.combo_IVtype.currentIndex()))+"\n")
        g.ser.write_b(str(int(self.combo_IVoption.currentIndex()))+"\n")
        g.ser.write_b(str(int(self.returnCheck))+"\n")

    def programOne(self):
        self.programDevs([[g.w, g.b]])

    def programRange(self):
        devs = makeDeviceList(True)
        self.programDevs(devs)

    def programAll(self):
        devs = makeDeviceList(False)
        self.programDevs(devs)

    def programDevs(self, devs):
        totalCycles = int(self.rightEdits[0].text())

        job="201"
        g.ser.write_b(job+"\n")   # sends the job

        self.sendParams()
        wrapper = ThreadWrapper(devs, totalCycles)
        self.execute(wrapper, wrapper.run)

    @staticmethod
    def display(w, b, raw, parent=None):

        resistance = []
        voltage = []
        current = []
        abs_current = []

        # Find nr of cycles
        lineNr = 1
        totalCycles = 0
        resistance.append([])
        voltage.append([])
        current.append([])
        abs_current.append([])

        resistance[totalCycles].append(raw[0][0])
        voltage[totalCycles].append(raw[0][1])
        current[totalCycles].append(raw[0][1]/raw[lineNr][0])
        abs_current[totalCycles].append(abs(current[totalCycles][-1]))


        # take all data lines without the first and last one (which are _s and
        # _e)
        while lineNr < len(raw)-1:
            currentRunTag = raw[lineNr][3]

            while (currentRunTag == raw[lineNr][3]):
                resistance[totalCycles].append(raw[lineNr][0])
                voltage[totalCycles].append(raw[lineNr][1])
                current[totalCycles].append(raw[lineNr][1]/raw[lineNr][0])
                abs_current[totalCycles].append(abs(current[totalCycles][-1]))

                lineNr += 1
                if lineNr == len(raw):
                    break
            totalCycles += 1
            resistance.append([])
            voltage.append([])
            current.append([])
            abs_current.append([])

        resistance[totalCycles - 1].append(raw[-1][0])
        voltage[totalCycles - 1].append(raw[-1][1])
        current[totalCycles - 1].append(raw[-1][1]/raw[-1][0])
        abs_current[totalCycles - 1].append(abs(current[totalCycles - 1][-1]))

        # setup display
        resultWindow = QtWidgets.QWidget()
        resultWindow.setGeometry(100,100,1000*g.scaling_factor,400)
        resultWindow.setWindowTitle("Curve Tracer: W="+ str(w) + " | B=" + str(b))
        resultWindow.setWindowIcon(Graphics.getIcon('appicon'))
        resultWindow.show()

        view=pg.GraphicsLayoutWidget()

        label_style = {'color': '#000000', 'font-size': '10pt'}


        plot_abs = view.addPlot()
        plot_abs.getAxis('left').setLabel('Current', units='A', **label_style)
        plot_abs.getAxis('bottom').setLabel('Voltage', units='V', **label_style)
        plot_abs.setLogMode(False, True)
        plot_abs.getAxis('left').setGrid(50)
        plot_abs.getAxis('bottom').setGrid(50)

        # go to next row and add the next plot
        view.nextColumn()

        plot_IV = view.addPlot()
        plot_IV.addLegend()
        plot_IV.getAxis('left').setLabel('Current', units='A', **label_style)
        plot_IV.getAxis('bottom').setLabel('Voltage', units='V', **label_style)
        plot_IV.getAxis('left').setGrid(50)
        plot_IV.getAxis('bottom').setGrid(50)

        # go to next row and add the next plot
        view.nextColumn()

        plot_R = view.addPlot()
        plot_R.getAxis('left').setLabel('Resistance', units='Ohms',
                **label_style)
        plot_R.getAxis('bottom').setLabel('Voltage', units='V', **label_style)
        plot_R.setLogMode(False, True)
        plot_R.getAxis('left').setGrid(50)
        plot_R.getAxis('bottom').setGrid(50)

        resLayout = QtWidgets.QVBoxLayout()
        resLayout.addWidget(view)
        resLayout.setContentsMargins(0,0,0,0)

        resultWindow.setLayout(resLayout)

        # setup range for resistance plot
        maxRes_arr = []
        minRes_arr = []

        for cycle in range(1, totalCycles + 1):
            maxRes_arr.append(max(resistance[cycle - 1]))
            minRes_arr.append(min(resistance[cycle - 1]))

        maxRes = max(maxRes_arr)
        minRes = max(minRes_arr)

        for cycle in range(1,totalCycles+1):
            aux1 = plot_abs.plot(pen=(cycle, totalCycles), symbolPen=None,
                    symbolBrush=(cycle, totalCycles), symbol='s', symbolSize=5,
                    pxMode=True, name='Cycle ' + str(cycle))
            aux1.setData(np.asarray(voltage[cycle - 1]),
                    np.asarray(abs_current[cycle - 1]))

            aux2 = plot_IV.plot(pen=(cycle, totalCycles), symbolPen=None,
                    symbolBrush=(cycle, totalCycles), symbol='s', symbolSize=5,
                    pxMode=True, name='Cycle ' + str(cycle))
            aux2.setData(np.asarray(voltage[cycle - 1]),
                    np.asarray(current[cycle - 1]))

            aux3 = plot_R.plot(pen=(cycle, totalCycles), symbolPen=None,
                    symbolBrush=(cycle, totalCycles), symbol='s', symbolSize=5,
                    pxMode=True, name='Cycle ' + str(cycle))
            aux3.setData(np.asarray(voltage[cycle - 1]),
                    np.asarray(resistance[cycle - 1]))

        plot_R.setYRange(np.log10(_min_without_inf(resistance, np.inf)),
                np.log10(_max_without_inf(resistance, np.inf)))
        plot_abs.setYRange(np.log10(_min_without_inf(abs_current, 0.0)),
                np.log10(_max_without_inf(abs_current, 0.0)))

        resultWindow.update()

        return resultWindow


g.DispCallbacks[tag] = CurveTracer.display
