import numpy as np
import matplotlib.pyplot as plt
import pyroomacoustics as pyra
import webrtcvad
class dataSetAudio:

    def __init__(self, doa_array="random", rt60_array="random",
                 SNR_array="random", dist_array="random", duracion=0.5):

        if type(doa_array)!=str:        
            self.doas = np.array([doa_array] if np.logical_or(type(doa_array)==int,type(doa_array)==float)
                                 else doa_array)
            self.doas = np.array([doa_array] if np.logical_or(type(doa_array)==np.int_,type(doa_array)==np.float_)
                                 else self.doas)
        else:
            self.doas = doa_array
            
        if type(rt60_array)!=str:
            self.rt60 = np.array([rt60_array] if np.logical_or(type(rt60_array)==int,type(rt60_array)==float)
                                  else rt60_array)
            self.rt60 = np.array([rt60_array] if type(rt60_array)==np.float_ else self.rt60)
        else:
            self.rt60 = rt60_array
            
        if type(SNR_array)!=str:
            self.snr = np.array([SNR_array] if np.logical_or(type(SNR_array)==int,type(SNR_array)==float)
                                else SNR_array)
            self.snr = np.array([SNR_array] if type(SNR_array)==np.float_ else self.snr)
        else:
            self.snr = SNR_array
            
        if type(dist_array)!=str:
            self.dist = np.array([dist_array] if np.logical_or(type(dist_array)==int,type(dist_array)==float)
                                 else dist_array)
            self.dist = np.array([dist_array] if type(dist_array)==np.float_ else self.dist)
        else:
            self.dist = dist_array
#         self.num = num
        self.duracion = duracion
        

    def Segmentar(self, data, fs, time):
        """to segment the signal into segments that can be passed to the VAD and other"""
        
        if type(data) is list:
            a = np.array([j for i in data for j in i])
        else:
            a=data
        num_muestras = int(fs*time)
        X = np.array([a[i:i+num_muestras] for i in range(0, len(a)-num_muestras, num_muestras)])
        return X

    def VadSeg(self, aggr, data, fs, time):
        """this function performs VADs on voice segments passed to it.
         aggr: is a parameter that determines the aggressiveness of the selection
         time in ms and should be [10,20,30 ms]"""

        vad = webrtcvad.Vad(aggr)

        tiempo = time/1000
        
        X = self.Segmentar(data, fs, tiempo)

        #get the indices where there is voice and create list of voice segments
        a_voz = list(map(lambda y: X[y], [i for i in range(len(X)) 
                                          if vad.is_speech(np.int16(X[i]).tobytes(), fs)]))

        
        A_voz = np.array([j for i in a_voz for j in i])

        return A_voz
    
    

    def SimuData(self, destino, data, fs, M, room_dim=np.r_[10., 10, 3], delay=[0], N=1, aggr=3, vad_ms=20,
                array_mic="ULA", dataset=True, plot=[False,3,0], length=(0,7200), split=True,random=0):
        """This function (class method), depending on the value of the "datset" flag, can generate a dataset
         file with blocks of duration segments set by the user (half a second by default), each block will
         have the characteristics set by the user"""
        
        if np.logical_and(type(self.doas)==str, array_mic=="ULA"):
            self.doas = np.random.randint(-90,90,random)
        elif np.logical_and(type(self.doas)==str, array_mic=="UCA"):
            self.doas = np.random.randint(0,360,random)
            
        if type(self.dist)==str:
            self.dist = np.random.random(random)*1.5+1.5
        elif self.dist.size != self.doas.size:
            multiplo = int(np.ceil(self.doas.size/self.dist.size))
            temp = self.dist.tolist()*multiplo
            self.dist = np.asarray(temp[:self.doas.size])
        
        if type(self.rt60)==str:
            temp = np.random.random(random)
            self.rt60 = np.where(temp<0.18, 0.18, temp)
        elif self.rt60.size != self.doas.size:
            multiplo = int(np.ceil(self.doas.size/self.rt60.size))
            temp = self.rt60.tolist()*multiplo
            self.rt60 = np.asarray(temp[:self.doas.size])
        
        if type(self.snr)==str:
            self.snr = np.random.random(random)*20
        elif self.snr.size != self.doas.size:
            multiplo = int(np.ceil(self.doas.size/self.snr.size))
            temp = self.snr.tolist()*multiplo
            self.snr = np.asarray(temp[:self.doas.size])

        st = int(length[0]*fs)
        ed = int(length[1]*fs)
        
        if dataset==True:
            
            datasetAudio = []
            
            DD = self.VadSeg(aggr,data,fs,vad_ms)[st:ed]
            
            if split:
                salto = int(self.duracion*fs)
                D = [DD[chunk-salto:chunk] for chunk in range(salto,DD.shape[-1]+1,salto)]
                np.random.shuffle(D)
            else:
                D = [DD]
         
            for chunk in D:
                for i,j,k,l in zip(self.doas,self.rt60,self.snr,self.dist):

                    d = 0.04287
                    lamb = 343/4000
                    rc = lamb/(4*np.sin(np.pi/M))

                    e_absorption, max_order = pyra.inverse_sabine(j, room_dim)

                    # noise variance    
                    sigma2 = 10 ** (-k / 10) / (4.0 * np.pi * l) ** 2

                    room = pyra.ShoeBox(room_dim, fs = fs, materials=pyra.Material(e_absorption),
                                        max_order=max_order, sigma2_awgn=sigma2)

                    if len(room_dim) == 2:   
                        if array_mic == "ULA":
                            Ang = (90-i)/180*np.pi
                            A = pyra.beamforming.linear_2D_array(room_dim / 2, M, 180/180*np.pi, d)
                        elif array_mic == "UCA":
                            Ang = ((360+i)-360*0**np.heaviside(-i,0))/180*np.pi
                            A = pyra.beamforming.circular_2D_array(room_dim / 2, M, 180/180*np.pi, rc)

                        s_locs = room_dim / 2 + l * np.r_[np.cos(Ang), np.sin(Ang)]
                        room.add_source(s_locs, signal=chunk, delay=delay[0])
                        room.add_microphone_array(pyra.MicrophoneArray(A, fs=room.fs))
                        pass

                    elif len(room_dim) == 3:
                        if array_mic == "ULA":
                            Ang = (90-i)/180*np.pi
                            A = pyra.beamforming.linear_2D_array(room_dim[:2] / 2, M, 180/180*np.pi, d)
                        elif array_mic == "UCA":
                            Ang = ((360+i)-360*0**np.heaviside(-i,0))/180*np.pi
                            A = pyra.beamforming.circular_2D_array(room_dim[:2] / 2, M, 180/180*np.pi, rc)

                        s_locs = room_dim / 2 + l * np.r_[np.cos(Ang), np.sin(Ang), 0]
                        room.add_source(s_locs, signal=chunk, delay=delay[0])
                        mics_array = np.array([[A[0][i1], A[1][i1], room_dim[-1]/2] for i1 in range(M)])
                        room.add_microphone_array(pyra.MicrophoneArray(mics_array.T,fs=room.fs))                                                   
                        pass

                    room.simulate()

                    info_list = [i,l,k,j,self.duracion,salto]   
                    list(map(lambda s: info_list.append(s), room.mic_array.signals[:,:salto].flatten().tolist()))
                    datasetAudio.append(info_list)
            
                f = open(destino,"a")
                np.random.shuffle(datasetAudio)
                for n in datasetAudio:
                    f.write(",".join(str(elto) for elto in n)+"\n")
                f.close()
                datasetAudio = []
                
                if np.logical_and(random,array_mic=="ULA"):
                    self.doas = np.random.randint(-90,90,self.doas.size)
                elif np.logical_and(random,array_mic=="UCA"):
                    self.doas = np.random.randint(0,360,self.doas.size)
                pass
            

        
        elif dataset==False:
            
            d = 0.04287
            lamb = 343/4000
            rc = lamb/(4*np.sin(np.pi/M))
            dist_l = self.dist.tolist()
            [dist_l.append(dist_l[-1]) for i in range(N-len(dist_l))]
            dist = np.array(dist_l)
            dataN = ([data] if N==1 else data)
            
            e_absorption, max_order = pyra.inverse_sabine(self.rt60[0], room_dim)

            # noise variance
            sigma2 = 10 ** (-self.snr[0] / 10) / (4.0 * np.pi * dist.max()) ** 2

            room = pyra.ShoeBox(room_dim, fs = fs, materials=pyra.Material(e_absorption), max_order=max_order,
                               sigma2_awgn=sigma2)

            if len(room_dim) == 2:
                if array_mic == "ULA":
                    for a in range(N):
                        Ang = (90-self.doas[a])/180*np.pi
                        s_locs = room_dim / 2 + dist[a] * np.r_[np.cos(Ang), np.sin(Ang)]
                        room.add_source(s_locs, signal=self.VadSeg(aggr,dataN[a],fs,vad_ms), delay=delay[a])
                    A = pyra.beamforming.linear_2D_array(room_dim / 2, M, 180/180*np.pi, d)
                elif array_mic == "UCA":
                    for a in range(N):
                        Ang = ((360+self.doas[a])-360*0**np.heaviside(-self.doas[a],0))/180*np.pi
                        s_locs = room_dim / 2 + dist[a] * np.r_[np.cos(Ang), np.sin(Ang)]
                        room.add_source(s_locs, signal=self.VadSeg(aggr,dataN[a],fs,vad_ms), delay=delay[a])
                    A = pyra.beamforming.circular_2D_array(room_dim / 2, M, 180/180*np.pi, rc)
                
                room.add_microphone_array(pyra.MicrophoneArray(A, fs=room.fs))
                pass

            elif len(room_dim) == 3:
                if array_mic == "ULA":
                    for a in range(N):
                        Ang = (90-self.doas[a])/180*np.pi
                        s_locs = room_dim / 2 + dist[a] * np.r_[np.cos(Ang), np.sin(Ang),0]
                        room.add_source(s_locs, signal=self.VadSeg(aggr,dataN[a],fs,vad_ms), delay=delay[a])
                    A = pyra.beamforming.linear_2D_array(room_dim[:2] / 2, M, 180/180*np.pi, d)
                elif array_mic == "UCA":
                    for a in range(N):
                        Ang = ((360+self.doas[a])-360*0**np.heaviside(-self.doas[a],0))/180*np.pi
                        s_locs = room_dim / 2 + dist[a] * np.r_[np.cos(Ang), np.sin(Ang), 0]
                        room.add_source(s_locs, signal=self.VadSeg(aggr,dataN[a],fs,vad_ms), delay=delay[a])
                    A = pyra.beamforming.circular_2D_array(room_dim[:2] / 2, M, 180/180*np.pi, rc)
                    
                
                mics_array = np.array([[A[0][i], A[1][i], room_dim[-1]/2] for i in range(M)])
                room.add_microphone_array(pyra.MicrophoneArray(mics_array.T,fs=room.fs))
                pass

            room.simulate()
            
            if plot[0]:
                fig1 = plt.figure()
                room.plot(img_order=plot[1])
                if plot[1] == 0:
                    plt.title("Posiciones del arreglo y las fuentes de voz en la habitación simulada")
                else:
                    plt.title("Número de reflexiones consideradas por ISM para calcular la reverberación")
                plt.ylabel("Largo [m]")
                plt.xlabel("Ancho [m]")
                fig2 = plt.figure()
                plt.plot(room.mic_array.signals[plot[2],:])
                fig3 = plt.figure()
                rir_1_0 = room.rir[1][0]
                plt.plot(np.arange(len(rir_1_0)) / room.fs, rir_1_0)
                plt.title("La RIR de la fuente al micrófono 1")
                plt.xlabel("Tiempo [s]")
#             IPython.display.Audio(room.mic_array.signals,rate=fs)
                return room.mic_array.signals[:,st:ed],fig1,fig2,fig3
            	
            return room.mic_array.signals[:,st:ed]
            	
            	
            	
            	
            
