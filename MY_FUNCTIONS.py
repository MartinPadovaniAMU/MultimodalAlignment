#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import random as random
import re
from scipy import stats
from scipy.signal import savgol_filter
from pathlib import Path


# In[2]:


def repare_time(string):
    
    """
    
    Repare a time point in string format with errors
    
    Args:
        Time point, in string format
        
    Returns:
        Time point, corrected, in string format

    """
    
    if type(string) != str:
        print("it is a string : ", string)
        string = str(string)
    index = [] #là où on mettra les ':'
    j=1
    while j<len(string)-2 and string[j+2]!='.':
        if j%2==1:
            index.append(j)
        j+=1
    for position in index[::-1]:
        string = string[:position+1] + ':' + string[position+1:]
    return string


# In[3]:


def conversion_time(strings):
    
    """
    
    Converts a time point from string format to float format
    
    Args:
        Time point, in string format
        
    Returns:
        Time point, in float format, in minuts
    
    """
    
    if type(strings)==str:
        
        string = strings
        if ':' not in string:
            string = repare_time(string)
        string = re.split(':', string)
        if len(string) >= 2:
            seconds_str, minuts_str = string[-1], string[-2]
            seconds, minuts = float(seconds_str), float(minuts_str)
            time = seconds/60 + minuts
            time_sec = time*60
        else:
            print("trop court pour séparer", string)
    elif (type(strings) == int or type(strings)== float) and strings <= 2000 :
        time = strings/60  #c'est en secondes
    elif (type(strings) ==int or type(strings) == float) and strings > 2000 : 
        strings = repare_time(strings)
        strings = conversion_time(strings)
        time = strings
    else:
        time=[]
        time_sec = []
        for string in strings :
            if type(string)==str:
                if ':' not in string:
                    string = repare_time(string)
                string = re.split(':', string)
                if len(string) >= 2:
                    seconds_str, minuts_str = string[-1], string[-2]
                    seconds, minuts = float(seconds_str), float(minuts_str)
                    time.append(seconds/60 + minuts)
                    time_sec.append(seconds + 60*minuts)
                else:
                    print("too short to separate")
            elif (type(string) == int or type(string)== float) and string <= 2000:
                time.append(string/60) #c'est en secondes
                time_sec.append(string)
            elif (type(string) == int or type(string)== float) and string > 2000:
                string = repare_time(string)
                string = conversion_time(string)
                time.append(string)
                time_sec.append(string*60)
            else:
                print(type(string))
    return time_sec


# In[4]:


def smooth(y, box_pts):
    
    """
    
    Smooths a function
    
    Args:
        Data to smooth, as a list, array or panda series
        Number of points for the smoothing window
        
    Returns:
        List of same length, smoothed data
        
    """
    
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# In[1]:


def smooth_2(y, window, poly):
    
    """
    
    Smooths a function
    
    Args:
        Data to smooth, as a list, array or panda series
        Number of points for the smoothing window
        Order of the polynome
        
    Returns:
        List of same length, smoothed data
        
    """
    
    return savgol_filter(y, window, poly)


# In[40]:


def smoothed_column(data, window, category):
    
    """
    
    Creates a list of smoothed data
    
    Args:
        DataFrame, with columns 'Filename', 'Speaker', 'Speech rate'
        window, in integer format
        
    Returns:
        List of smoothed data, same length
        
    """
    
    if category == 'AA':
        speaker1 = 'Adult1'
        speaker2 = 'Adult2'
    else:
        speaker1 = 'Parent'
        speaker2 = 'Child' 
        
    if len(data[data['Speaker']==speaker1])!=0 and len(data[data['Speaker']==speaker2])!=0:
        smooth1 = smooth(data[data['Speaker']==speaker1]['Speech rate'], window)
        smooth2 = smooth(data[data['Speaker']==speaker2]['Speech rate'], window)

        i1 = 0
        i2 = 0

        smooth_speech_rate = []

        for speaker in data['Speaker']:
            if speaker == speaker1:
                smooth_speech_rate.append(smooth1[i1])
                i1 += 1
            else:
                smooth_speech_rate.append(smooth2[i2])
                i2 += 1
                
    #on met les valeurs de smooth speech rate dans l'ordre
                
    elif len(data[data['Speaker']==speaker1])!=0 and len(data[data['Speaker']==speaker2])==0:
        smooth1 = smooth(data[data['Speaker']==speaker1]['Speech rate'], window)
        smooth_speech_rate = smooth1
    
    elif len(data[data['Speaker']==speaker1])==0 and len(data[data['Speaker']==speaker2])!=0:
        smooth2 = smooth(data[data['Speaker']==speaker2]['Speech rate'], window)
        smooth_speech_rate = smooth2
        
    else:
        print(speaker1, data['Speaker'][0], speaker1 == data['Speaker'][0], type(speaker1), type(data['Speaker'][0]))
        print('neither speaker1 nor speaker2 are in the Speakers')
         
    return smooth_speech_rate  


# In[6]:


def choose_color(label):
    

    """
    
    Defines the color for each label
    
    Args:
        Label, in a string format
        
    Returns:
        Color, in a string format
        
    """
    
    if "PREGAME" in label : 
        return "yellow"
    elif "POSTGAME" in label:
        return 'orange'
    elif "GAME" in label : 
        return "green"
    else:
        return "black"


# In[7]:


def arrondi(array, n):
    
    """
    
    Round up the array's elements to the inferior n multiple
    
    Args:
        Array of numbers
        Integer n
        
    Returns:
        Array of rounded up numbers
        
    """
    
    array = np.vectorize(float)(array)
    return array - array%n


# In[8]:


def find_best_arrondi(Filename, Global_start, smooth_speech_rate, Speaker):
    
    """
    
    Finds the smallest integer to round up time, in order to have an utterance of each speaker in each time interval
    
    Args:
        Filename : list, array, panda series of filenames for each utterance)
        Global_start : list,array, panda_series of starting times for each utterance
        Smooth_speech_rate : smoothed value of the speech rate for each utterance
        Speaker : speaker for each utterance
        
    Returns:
        Pivot table of the data grouped by time intervals, columns 'smooth_speech_rate, 'round tim', 'Speaker'
        Integer for the round up
        
    """
    
    local_data = pd.DataFrame(list(zip(Filename, Global_start, smooth_speech_rate,Speaker)), columns = ['Filename', 'Global_start', 'smooth_speech_rate', 'Speaker'])
    filename = local_data['Filename'][0]
    category = filename[0:2]
    if category == 'AA':
        speaker1 = 'Adult1'
        speaker2 = 'Adult2'
    else:
        speaker1 = 'Parent'
        speaker2 = 'Child'
    for n in range(1, int(max(local_data['Global_start']))):
        local_data['round time'] = arrondi(Global_start, n)
        tableau = local_data.pivot_table(values = 'smooth_speech_rate', index = 'round time', columns = 'Speaker')
        if (tableau.isna().any()[speaker1] == False and tableau.isna().any()[speaker2] == False) :
            return local_data.pivot_table(values = 'smooth_speech_rate', index = 'round time', columns = 'Speaker'), n


# In[9]:


def find_phases(times, phases_times):
    
    """
    
    Finds the three phases (postgame, game, postgame)
    
    Args:
        times : list of all time points in the conversation
        phases_times : list of couples [phase label, time]
        
    Returns:
        List of phases, in the ordre of the times. Can be added to the DataFrame
    
    """
    
    phases = []
    
    time_game = -1
    time_postgame = -1
    
    for couple in phases_times :
        
        label,time = couple
        
        if "POSTGAME" in label:
            time_postgame = time
        elif "GAME" in label :
            time_game = time
        
    time = times[0]
    i=0
    
    while time < time_game :
        phases.append("PREGAME")
        i+=1
        time = times[i]
        
    while time < time_postgame:
        phases.append("GAME")
        i+=1
        time = times[i]
    
    phases += ["POSTGAME"]*(len(times)-len(phases))
                
    return phases


# In[10]:


def find_roles(times, roles_times):
    
    """
    
    Finds the roles (speaker1 or speaker2)
    
    Args:
        times : list of all time points in the conversation
        roles_times : list of couples [speaker, time]
        
    Returns:
        List of roles, in the ordre of the times. Can be added to the DataFrame
    
    """
      
    
    roles = []
    
    j=0 
    couple = roles_times[0]
    
    while "SPEAKER" not in couple[0]:
        
        j+=1
        couple = roles_times[j]
    
    #On a trouvé le premier tour d'un des speakers
    
    label, start_time = couple
    
    i=0
    time = times[0]
    while time < start_time:
        i+=1
        time = times[i]
        roles.append(0)
    
    j+=1
    couple = roles_times[j]

    next_label, end_time = couple
    
    while "SPEAKER" in label and j < len(roles_times)-1:        
        
        while time < end_time :
            i+=1
            time = times[i]
            roles.append(label)
        
        label = next_label
        
        j += 1
        couple = roles_times[j]
        next_label, end_time = couple
        
    while time < end_time : 
        i+=1
        time = times[i]
        roles.append(label)

    roles += [0]*(len(times)-len(roles))

    return roles


# In[11]:


def label_intervals(values, limits):
    
    """
    Part a series of values into several intervals, according to limits
    
    Args:
        values : list, array of values
        limits : list, array of limits
        
    Returns :
        List of intervals (labeled with numbers) in the order of the values
        
    """
    
    intervals = []
    for value in values :
        if value < limits[0]:
            intervals.append(0)
        else:
            if len(limits)==1:
                intervals.append(1)
            elif len(limits)>1:
                i=1
                limit = limits[1]
                while value >= limit and i<len(limits)-1 :
                    i += 1
                    limit = limits[i]
                intervals.append(i)
                
    return intervals


# In[12]:


def fusion(data1, data2, col_order, col_values):
    
    """
    
    Creates a new list out of two lists, by putting the elements in the right order
    
    Args:
        data1 : array or panda, with columns col_order and col_values
        data2 : same
        col_order : string format, name of the column that gives the order
        col_values : string format, name of the column with the values to order
        
    Returns:
        data : a DataFrame, with two columns
        
    """
    
    data1.reset_index(inplace=True, drop=True)
    data2.reset_index(inplace=True, drop=True)
    
    i1, i2 = 0, 0
    index, values = [],[]
    
    while i1 < len(data1) and i2 < len(data2) :
        
        situation = int( data2[col_order][i2] < data1[col_order][i1]) #0 si data1 plus petite valeur, 1 si data2
        
        index.append( situation*data2[col_order][i2] + (1-situation)*data1[col_order][i1] )
        values.append( situation*data2[col_values][i2] + (1-situation)*data1[col_values][i1] )
        
        i1 += 1-situation
        i2 += situation 
        
    for i in range(i1, len(data1)):
        index.append(data1[col_order][i])
        
    for value in data1[i1:]:
        values.append(value)
        
    for i in range(i2, len(data2)):
        index.append(data2[col_order][i])
    
    for value in data2[i2:]:
        values.append(value)
        
    df = pd.DataFrame(list(zip(index, values)), columns=[col_time, col_values])
    
    return df


# In[13]:


def adapt_points(data, col_speaker, col_time, col_SR):
    
    """
    
    Adapts one of the DataFrames in order to have the same number of points in both of them. 
    Chooses the shorter one, defines the limits as all its values
    Groups the second one by intervals between those values
    
    Args:
        data1 : DataFrame with only speaker1, columns col_time and the speech rate
        data2 : DataFrame with only speaker2, columns col_time and the speech rate
        col_SR : name of the column used for speech rate ('Speech rate', or 'smoothed'), string format
        
    Returns:
        Panda serie of speaker1's speech rate
        Panda serie of speaker2's speech rate (of same length)
        
    """
    
    speakers = np.unique(data[col_speaker])
    speaker1, speaker2 = speakers[0], speakers[1]
    
    data1 = data[ data[col_speaker] == speaker1 ][[col_time, col_SR]]
    data2 = data[ data[col_speaker] == speaker2 ][[col_time, col_SR]]
    
    len1, len2 = len(data1), len(data2)
    
    if len1 == 1 or len2 == 1 :
        
        return [np.mean(data1[col_SR])], [np.mean(data2[col_SR])]
    
    elif len1 < len2 : 
        
        frontieres = [1/2*(list(data1[col_time])[i+1] + list(data1[col_time])[i]) for i in range(len(data1)-1)]
        intervals = label_intervals(data2[col_time], frontieres)
        
        data2['interval'] = intervals
        table2 = data2.pivot_table(values = [col_SR, col_time], index="interval", aggfunc=[np.mean, np.mean])
        
        data = fusion(data1, table2, col_time, col_SR)
        
        return data
    
    elif len2 < len1 : 
        
        frontieres = [1/2*(list(data2[col_time])[i+1] + list(data2[col_time])[i]) for i in range(len(data2)-1)]

        intervals = label_intervals(data1[col_time], frontieres)
        data1['interval'] = intervals
        table1 = data1.pivot_table(values = [col_SR, col_time], index="interval")
        
        data = fusion(table1, data2, col_time, col_SR)
        
        return data
    
    else:
        
        data = fusion(data1, data2, col_time, col_SR)
        
        return data


# In[14]:


def estimate_syll(word):  
    
    """
    
    Estimates the number of syllables of a french word that is not in the lexicon, even if some vowels with accents are replaced by "??"
    
    Args:
        Word, in string format
        
    Returns:
        Number of syllables, in integer format
        
    """
    
    word=word.lower()
    if '??' in word:
        caracteres = ['à', 'é', 'è', 'ç', 'ù', 'ê', 'û', 'ô', 'ï', 'î']
        for caractere in caracteres : 
            possible_word = word.replace("??", caractere)
            if possible_word in dictionnary:
                return dictionnary[possible_word]
        return 1
                
    else:
        vowels = ['a', 'e', 'é', 'è', 'à','i', 'o', 'u', 'y']
        nb = 0
        for character in word:
            if character in vowels:
                nb+=1
    return nb


# In[15]:


def continuous_version(times_list, data_list, t):
    
    """
    
    Computes the value of a function at any point, when we only have a few data points for the function
    
    Args:
        times_list : list or array, timepoints for the function
        data_list : list or array, values corresponding to the timepoints
        t : timepoint at which we want to compute the function
        
    Returns:
        Value, in float format
        
    """
    
    i=0
    time = times_list[i]
    while times_list[i+1]<t and i<len(times_list)-1:
        i+=1
        time = times_list[i+1]
    value = data_list[i] + (t-times_list[i])*(data_list[i+1]-data_list[i])/(times_list[i+1]-times_list[i])
    
    return value


# In[16]:


def who_speaks(data, i, col_speaker):
    
    """
    
    Checks how many utterances each speaker has left (after line i, excluded)
    
    Args:
        data : dataFrame, with column 'Speaker'
        index : index from which we look for the speakers
        
    Returns:
        Nb of utterances of speaker1
        Nb of utterances of speaker2
        
    """
    speaker1, speaker2 = np.unique(data[col_speaker])
    
    end_data = data[i:]
    nb_speaker1 = len(end_data[end_data[col_speaker]==speaker1])
    nb_speaker2 = len(end_data[end_data[col_speaker]==speaker2])
    
    return np.array([nb_speaker1, nb_speaker2])


# In[17]:


def adapt_points_2(data, col_speaker, col_time, col_SR):
    
    """
    
    Modifies the table in order to have the same number of points for both speakers by averaging some of them, but without reducing too much the number of points
    
    Args:
        data : DataFrame, with time, speaker and speech rate columns
        col_speaker : string, name of the speakers column
        col_time : string, name of the time column
        col_SR : string, name of the speech rate column
        
    Returns:
        DataFrame, with a smaller number of lines, with as many speaker1 utterances than speaker2
        
        """
    
    
    data.reset_index(drop=True, inplace=True)
    
    
    new_time, new_speaker, new_SR = [], [], []
    
    current_speaker = data[col_speaker][0]
    current_SR, current_nb, current_time = data[col_SR][0], 1, data[col_time][0]
    
    i=1
    
    while i < len(data[col_speaker])-2:
        
        
        if data[col_speaker][i] != current_speaker:
            
            
            new_time.append(current_time/current_nb)
            new_speaker.append(current_speaker)
            new_SR.append(current_SR/current_nb)
            
            current_speaker = data[col_speaker][i]
            current_SR, current_nb, current_time = data[col_SR][i], 1, data[col_time][i]
            
        else:
            
            current_SR += data[col_SR][i]
            current_nb += 1
            current_time += data[col_time][i]
            
        i += 1
        
    speaker1, speaker2 = np.unique(new_speaker)
    n1, n2 = who_speaks(data, i-1, speaker1, speaker2, col_speaker)
    
    if new_speaker.count(speaker1)+n1 == new_speaker.count(speaker2) + n2 :
        
        if data[col_speaker][i] != current_speaker:
            
            new_time.append(current_time/current_nb)
            new_speaker.append(current_speaker)
            new_SR.append(current_SR/current_nb)

            current_speaker = data[col_speaker][i]
            current_SR, current_nb, current_time = data[col_SR][i], 1, data[col_time][i]
            i+=1
            
        else:
            
            current_SR += data[col_SR][i]
            current_nb += 1
            current_time += data[col_time][i]
            
            i+=1
            
            
        if data[col_speaker][i] != current_speaker : 
            
            new_time.append(current_time/current_nb)
            new_speaker.append(current_speaker)
            new_SR.append(current_SR/current_nb)

            current_speaker = data[col_speaker][i]
            current_SR, current_nb, current_time = data[col_SR][i], 1, data[col_time][i]
            
        else:
            
            current_SR += data[col_SR][i]
            current_nb += 1
            current_time += data[col_time][i]
            
        new_time.append(current_time/current_nb)
        new_speaker.append(current_speaker)
        new_SR.append(current_SR/current_nb)
        
    else:
        
        remaining_data = data[i-3:]
        
        SR_1 = np.mean(remaining_data[remaining_data[col_speaker]==speaker1][col_SR])
        SR_2 = np.mean(remaining_data[remaining_data[col_speaker]==speaker2][col_SR])
        
        time_1 = np.mean(remaining_data[remaining_data[col_speaker]==speaker1][col_time])
        time_2 = np.mean(remaining_data[remaining_data[col_speaker]==speaker2][col_time])
        
        nb_1 = len(remaining_data[remaining_data[col_speaker]==speaker1])
        nb_2 = len(remaining_data[remaining_data[col_speaker]==speaker2])
        
        new_time.append(time_1/nb_1)
        new_speaker.append(speaker1)
        new_SR.append(SR_1/nb_1)
        
        new_time.append(time_2/nb_2)
        new_speaker.append(speaker2)
        new_SR.append(SR_2/nb_2)
            
    dictionary = {"Time":new_time, "Speaker":new_speaker, "Speech rate":new_SR}
    
    return pd.DataFrame(dictionary)


# In[37]:


def both_speak(data, i_start, i_end, col_speaker):
    
    """
    
    Checks if both speakers speak in the group of utterances
    
    Args:
        data
        i_start : first line of the group of utterances in data
        i_end : last line (not included)
        
    Returns : 
        Boolean : True if both speakers speak
        
    """
    
    speaker1, speaker2 = np.unique(data[col_speaker])
    return (speaker1 in list(data[col_speaker][i_start:i_end]) and speaker2 in list(data[col_speaker][i_start:i_end]))
    


# In[67]:


def adapt_points_3(data, col_speaker, col_time, col_SR):
    
    """
    
    Modifies the table in order to have the same number of points for both speakers by averaging some of them, but without reducing too much the number of points
    Updated version with the lists instead of array
    
    Args:
        data : DataFrame, with time, speaker and speech rate columns
        col_speaker : string, name of the speakers column
        col_time : string, name of the time column
        col_SR : string, name of the speech rate column
        
    Returns:
        DataFrame, with a smaller number of lines, with as many speaker1 utterances than speaker2, and columns speaker, time and speech rate
        
        """
    
    if len(np.unique(data[col_speaker]))==1:

        return data[[col_speaker, col_SR, col_time]]
        
    else:
        
        speaker1, speaker2 = np.unique(data[col_speaker])

        #clean_data = np.array([])
        clean_time = []
        clean_speaker = []
        clean_SR = []

        i_start = 0
        i_end = 1

        while both_speak(data, i_end, len(data),col_speaker) and i_end-1 < len(data) :
            

            if both_speak(data, i_start, i_end, col_speaker):
                

                table = data[i_start:i_end].pivot_table(values=[col_time, col_SR], index=col_speaker)
                table.reset_index(drop=False, inplace=True)
                


                if len(clean_time) == 0:
                    
                    
                    clean_time = list(table[col_time])
                    
                    clean_speaker = list(table[col_speaker])
                    
                    clean_SR = list(table[col_SR])

                else:
                    

                    clean_time += list(table[col_time])
                    clean_speaker += list(table[col_speaker])
                    clean_SR += list(table[col_SR])
                    

                i_start = i_end
                i_end += 1

            else :

                i_end += 1

        if i_end-1 != len(data):

            table = data[i_start:].pivot_table(values=[col_time, col_SR], index=col_speaker)
            table.reset_index(inplace=True, drop=False)

            if len(clean_time) == 0:
                clean_time = list(table[col_time])
                clean_speaker = list(table[col_speaker])
                clean_SR = list(table[col_SR])

            else:

                clean_time += list(table[col_time])
                clean_speaker += list(table[col_speaker])
                clean_SR += list(table[col_SR])
                
                
        dico = {col_time:clean_time, col_speaker:clean_speaker, col_SR:clean_SR}
        result = pd.DataFrame(dico)
        return result


# In[2]:


def compute_corr_phases(df, Phases, col_phases, col_values, col_speaker, col_time):
    
    """
    
    Computes the correlation of the two speakers' speech rates for each phase of the conversation, and the number of utterances points for each phase
    
    Args:
        df : DataFrame of the conversation
        Phases : List of the phases' names (string)
        col_values : name of the speech rate column, string format
        col_speaker : name of the speaker column, string format
        col_time : name of the time column, string format
        
    Returns:
        correlations : A list of None or arrays containing a diagonal of 1 and the wanted correlation on the other corners
        nb_points : A list of the number of points for the correlation computation for each phase
    """
    
    correlations = []
    nb_points = []
    p_values = []
    
    for phase in Phases :
        
        sub_df = df[ df[col_phases] == phase]
        nb = len(sub_df)
        
        clean_sub_df = adapt_points_3(sub_df, col_speaker, col_time, col_values)
        
        if len(np.unique(sub_df[col_speaker]))>1:
            
            speaker1, speaker2 = np.unique(sub_df[col_speaker])
        
            clean_sub_df_1, clean_sub_df_2 = clean_sub_df[ clean_sub_df[col_speaker]==speaker1], clean_sub_df[ clean_sub_df[col_speaker]==speaker2]
            
            if len(clean_sub_df_1)<2:
                corr, p_value = None,None
                
            else:
                corr, p_value = stats.pearsonr(list(clean_sub_df_1[col_values]), list(clean_sub_df_2[col_values]))
            
            correlations.append(corr)
            p_values.append(p_value)
            
        else:
            correlations.append(None)
            p_values.append(None)
        
        nb_points.append(nb)
        
    return correlations, nb_points, p_values


# In[1]:


def my_mean(my_list):
    
    """
    
    Computes the mean of a list containing None elements
    
    Args:
        my_list : the list
        
    Returns:
        mean : float type
        
    """
    
    mean, nb = 0,0
    
    for element in my_list :
        
        if element != None:
            mean += element
            nb +=1
 
    mean /= nb

    return mean         


# In[1]:


def compute_corr_roles(data, col_roles, col_values, col_speaker, col_time):
    
    
    """Returns a list of roles and a list of corresponding correlations"""
    
    data = data[data['Roles'] != 0]
    data = data[data["Roles"] != "0"]
    data.reset_index(inplace=True, drop=True)
    
    speaker1, speaker2 = np.unique(list(data[col_speaker]))
    
    roles = []
    correlations = []
    p_values = []
    
    current_role = data[col_roles][0]
    i_start = 0
    i_end = 1
    
    while i_end < len(data)-1 :
        
        if data[col_roles][i_end] == current_role :
            
            i_end +=1
            
        else :
            
            clean_data = adapt_points_3(data[i_start: i_end], col_speaker, col_time, col_values)
            
            
            data1 = clean_data[clean_data['Speaker']==speaker1]
            data2 = clean_data[clean_data['Speaker']==speaker2]
            
            if len(data1)==len(data2):
                
                correlation, p_value = stats.pearsonr(list(data1[col_values]), list(data2[col_values]))
                
            else:
                correlation = None
            
            correlations.append(correlation)
            p_values.append(p_value)
            
            short_role = current_role[-2:]
            
            while short_role in roles :
                short_role += "."
                
            roles.append(short_role)
            
            i_start = i_end
            i_end = i_end + 1
            current_role = data[col_roles][i_end]
            
    clean_data = adapt_points_3(data[i_start:i_end+1], col_speaker, col_time, col_values)
            
    data1 = clean_data[clean_data['Speaker']==speaker1]
    data2 = clean_data[clean_data['Speaker']==speaker2]
            
    correlation, p_value = stats.pearsonr(list(data1[col_values]), list(data2[col_values]))
    
    correlations.append(correlation)
    p_values.append(p_value)
    
    short_role = current_role[-2:]
    while short_role in roles :
        short_role += "."
                
    roles.append(short_role)
        
    return roles, correlations, p_values

