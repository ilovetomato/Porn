__author__ = 'yizhangzc'

import os
import shutil

import numpy as np
from absl import app, flags
from scipy import stats

from pandas import Series
from sliding_window import sliding_window

import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
from scipy.interpolate import interp1d
from scipy.fftpack import fft

"""
# for opp
tf.app.flags.DEFINE_integer( 'length', 300, 'the length of time window.' )
tf.app.flags.DEFINE_integer( 'overlap', 30, 'the overlap between time windows.' )
"""

# for pamap2
tf.app.flags.DEFINE_integer( 'length', 200, 'the length of time window.' )
tf.app.flags.DEFINE_integer( 'overlap', 100, 'the overlap between time windows.' )

def divide_in_sub_intervals(sensor_readings, sub_interval_length):
    sub_intervals = []
    for i in range(0, len(sensor_readings), sub_interval_length):
        if len(sensor_readings[i:i+sub_interval_length]) > 1:
            sub_intervals.append(sensor_readings[i:i+sub_interval_length, :])
    return sub_intervals
    
def append_fft_values_to_output(output, fft_values):
    for fft_value in fft_values: # fft_value.shape = (3,), fft_values.shape = (10, 3)
        for elem in fft_value:
            output.append(elem.real)
            output.append(elem.imag)
            
def get_sub_interval_fft(sub_interval):
    measurements = sub_interval # (10, 27)
    out = []
    for i in range(0, measurements.shape[1]-1, 3):
        current_sensor_measurements = measurements[:, i:i+3]     # (10, 3)
        fft_values = fft(current_sensor_measurements) # (10, 3)
        out.append(fft_values)
    return out # out is a list with 9 np.array (one per sensor) of complex values with shape (10, 3) 
    
def preprocess_and_get_output_list(sub_intervals):
    output = []
    for sub_interval in sub_intervals: # sub_interval.shape = (10, 27)
        sensors_fft_values = get_sub_interval_fft(sub_interval) #  (10, 3) * 9
        for sensor_values in sensors_fft_values:
            append_fft_values_to_output(output, sensor_values)
    return output

def preprocess_opportunity( ):

    #dataset_path    = '/data/zhangyi/uahar/opportunity/'
    dataset_path    = ''
    channel_num     = 113

    file_list = [   ['OpportunityUCIDataset/dataset/S1-Drill.dat',
                    'OpportunityUCIDataset/dataset/S1-ADL1.dat',
                    'OpportunityUCIDataset/dataset/S1-ADL2.dat',
                    'OpportunityUCIDataset/dataset/S1-ADL3.dat',
                    'OpportunityUCIDataset/dataset/S1-ADL4.dat',
                    'OpportunityUCIDataset/dataset/S1-ADL5.dat'] ,
                    ['OpportunityUCIDataset/dataset/S2-Drill.dat',
                    'OpportunityUCIDataset/dataset/S2-ADL1.dat',
                    'OpportunityUCIDataset/dataset/S2-ADL2.dat',
                    'OpportunityUCIDataset/dataset/S2-ADL3.dat',
                    'OpportunityUCIDataset/dataset/S2-ADL4.dat',
                    'OpportunityUCIDataset/dataset/S2-ADL5.dat'] ,
                    ['OpportunityUCIDataset/dataset/S3-Drill.dat',
                    'OpportunityUCIDataset/dataset/S3-ADL1.dat',
                    'OpportunityUCIDataset/dataset/S3-ADL2.dat',
                    'OpportunityUCIDataset/dataset/S3-ADL3.dat',
                    'OpportunityUCIDataset/dataset/S3-ADL4.dat',
                    'OpportunityUCIDataset/dataset/S3-ADL5.dat'] ,
                    ['OpportunityUCIDataset/dataset/S4-Drill.dat',
                    'OpportunityUCIDataset/dataset/S4-ADL1.dat',
                    'OpportunityUCIDataset/dataset/S4-ADL2.dat',
                    'OpportunityUCIDataset/dataset/S4-ADL3.dat',
                    'OpportunityUCIDataset/dataset/S4-ADL4.dat',
                    'OpportunityUCIDataset/dataset/S4-ADL5.dat'] ]

    invalid_feature = np.arange( 46, 50 )
    invalid_feature = np.concatenate( [invalid_feature, np.arange(59, 63)] )
    invalid_feature = np.concatenate( [invalid_feature, np.arange(72, 76)] )
    invalid_feature = np.concatenate( [invalid_feature, np.arange(85, 89)] )
    invalid_feature = np.concatenate( [invalid_feature, np.arange(98, 102)] )
    invalid_feature = np.concatenate( [invalid_feature, np.arange(134, 244)] )
    invalid_feature = np.concatenate( [invalid_feature, np.arange(245, 249)] )

    lower_bound = np.array([    3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                                3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                                3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                                3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,   3000,
                                3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                                3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                                3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                                3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                                3000,   3000,   3000,   10000,  10000,  10000,  1500,   1500,   1500,
                                250,    25,     200,    5000,   5000,   5000,   5000,   5000,   5000,
                                10000,  10000,  10000,  10000,  10000,  10000,  250,    250,    25,
                                200,    5000,   5000,   5000,   5000,   5000,   5000,   10000,  10000,
                                10000,  10000,  10000,  10000,  250, ])

    upper_bound = np.array([    -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                                -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                                -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                                -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,  -3000,
                                -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                                -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                                -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                                -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                                -3000,  -3000,  -3000,  -10000, -10000, -10000, -1000,  -1000,  -1000,
                                -250,   -100,   -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,
                                -10000, -10000, -10000, -10000, -10000, -10000, -250,   -250,   -100,
                                -200,   -5000,  -5000,  -5000,  -5000,  -5000,  -5000,  -10000, -10000,
                                -10000, -10000, -10000, -10000, -250, ])

    if os.path.exists( dataset_path + 'processed_data/' ):
        shutil.rmtree( dataset_path + 'processed_data/' )
    os.mkdir( dataset_path + 'processed_data/' )

    for usr_idx in range( 4 ):
        
        # import pdb; pdb.set_trace()
        print( "process data... user{}".format( usr_idx ) )
        time_windows    = np.empty( [0, FLAGS.length, channel_num], dtype=np.float )
        act_labels      = np.empty( [0], dtype=np.int )

        for file_idx in range( 6 ):

            filename = file_list[ usr_idx ][ file_idx ]

            file    = dataset_path + filename
            signal  = np.loadtxt( file )
            signal  = np.delete( signal, invalid_feature, axis = 1 )

            data    = signal[:, 1:114].astype( np.float )
            label   = signal[:, 114].astype( np.int )

            label[ label == 0 ] = -1

            label[ label == 101 ] = 0
            label[ label == 102 ] = 1
            label[ label == 103 ] = 2
            label[ label == 104 ] = 3
            label[ label == 105 ] = 4

            # label[ label == 406516 ] = 0
            # label[ label == 406517 ] = 1
            # label[ label == 404516 ] = 2
            # label[ label == 404517 ] = 3
            # label[ label == 406520 ] = 4
            # label[ label == 404520 ] = 5
            # label[ label == 406505 ] = 6
            # label[ label == 404505 ] = 7
            # label[ label == 406519 ] = 8
            # label[ label == 404519 ] = 9
            # label[ label == 406511 ] = 10
            # label[ label == 404511 ] = 11
            # label[ label == 406508 ] = 12
            # label[ label == 404508 ] = 13
            # label[ label == 408512 ] = 14
            # label[ label == 407521 ] = 15
            # label[ label == 405506 ] = 16

            # fill missing values using Linear Interpolation
            data    = np.array( [Series(i).interpolate(method='linear') for i in data.T] ).T
            data[ np.isnan( data ) ] = 0.

            # normalization
            diff = upper_bound - lower_bound
            data = ( data - lower_bound ) / diff

            data[ data > 1 ] = 1.0
            data[ data < 0 ] = 0.0

            #sliding window
            data    = sliding_window( data, (FLAGS.length, channel_num), (FLAGS.overlap, 1) ) 
            label   = sliding_window( label, FLAGS.length, FLAGS.overlap )
            label   = stats.mode( label, axis=1 )[0][:,0] 

            #remove non-interested time windows (label==-1)
            invalid_idx = np.nonzero( label < 0 )[0]
            data        = np.delete( data, invalid_idx, axis=0 ) 
            label       = np.delete( label, invalid_idx, axis=0 )

            time_windows    = np.concatenate( (time_windows, data), axis=0 )
            act_labels      = np.concatenate( (act_labels, label), axis=0 )

        np.save( dataset_path + 'processed_data/' + 'sub{}_features'.format( usr_idx ), time_windows )
        np.save( dataset_path + 'processed_data/' + 'sub{}_labels'.format( usr_idx ), act_labels )                
        print( "sub{} finished".format( usr_idx) )


def preprocess_pamap2_ds( ):
    dataset_path    = ''
    channel_num     = 27
    
    if os.path.exists( dataset_path + 'processed_data_ds/' ):
        shutil.rmtree( dataset_path + 'processed_data_ds/' )
    os.mkdir( dataset_path + 'processed_data_ds/' )

    lowerBound = np.array( [
        -18.1809,       -10.455566,     -7.7453649,         
        -3.9656347,     -2.54338,       -4.6695066,
        -40.503183,     -71.010566,     -68.132566,
        
        -4.0144814,     -2.169227,      -10.6296,       
        -1.0943914,     -1.7640771,     -0.85873557,    
        -35.379757,     -67.172728,     -42.7236,
        
        -1.540593,      -15.741095,     -12.220085,          
        -3.2952385,     -2.2376485,     -4.7753095,
        -92.0835,       -54.226,        -38.719725] )

    upperBound = np.array( [
        8.4245698,      18.942083,      10.753683,             
        4.0320432,      2.9766798,      4.4932249, 
        68.037749,      41.263183,      28.875164,
        
        4.8361714,      21.236957,      9.54976,                 
        1.13992,        1.7279,         0.84088028, 
        40.9678,        15.5543,        53.282157,
        
        26.630485,      24.1561,        6.9268395,              
        3.505307,       1.5628585,      6.4969625,
        5.87456,        44.9736,        61.16018] )

    file_list = [
        'subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat',
        'subject105.dat', 'subject106.dat', 'subject107.dat', 'subject108.dat'  ]

    for usr_idx in range(len(file_list)):

        file    = dataset_path + 'Protocol/' + file_list[usr_idx]
        data    = np.loadtxt( file ) # (376417, 54)

        label   = data[:,1].astype( int )
        label[label == 0]   = -1
        label[label == 1]   = 0         # lying
        label[label == 2]   = 1         # sitting
        label[label == 3]   = 2         # standing
        label[label == 4]   = 3         # walking
        label[label == 5]   = 4         # running
        label[label == 6]   = 5         # cycling
        label[label == 7]   = 6         # nordic walking
        label[label == 12]  = 7         # ascending stairs
        label[label == 13]  = 8         # descending stairs
        label[label == 16]  = 9         # vacuum cleaning
        label[label == 17]  = 10        # ironing
        label[label == 24]  = 11        # rope jumping

        # fill missing values
        #valid_idx   = np.concatenate( (np.arange(4, 16), np.arange(21, 33), np.arange(38, 50)), axis = 0 )
        valid_idx   = np.concatenate( (np.arange(4, 7), np.arange(10, 16), np.arange(21, 24),np.arange(27, 33), np.arange(38, 41), np.arange(44, 50)), axis = 0 )
        data        = data[ :, valid_idx ] # (376417, 27)
        data        = np.array( [Series(i).interpolate() for i in data.T] ).T

        # min-max normalization
        diff = upperBound - lowerBound
        data = 2 * (data - lowerBound) / diff - 1

        data[ data > 1 ]    = 1.0
        data[ data < -1 ]   = -1.0

        # sliding window
        data    = sliding_window( data, (FLAGS.length, channel_num), (FLAGS.overlap, 1) ) # (3763, 200, 27)
        label   = sliding_window( label, FLAGS.length, FLAGS.overlap )
        label   = stats.mode( label, axis=1 )[0][:,0]

        # remove non-interested time windows (label==-1)
        invalid_idx = np.nonzero( label < 0 )[0]
        data        = np.delete( data, invalid_idx, axis=0 )   # (2503, 200, 27)
        label       = np.delete( label, invalid_idx, axis=0 )  # (2503,)
        
        ''' deepsense like data process'''
        data_ds  = np.empty( [0, 20, 540], dtype=np.float )
        for i in range(len(data)):
            sample = data[i] # (200, 27)
            sub_intervals = divide_in_sub_intervals(sample, 10) # (10, 27) * 20
            output = preprocess_and_get_output_list(sub_intervals) # len(output)=10800
            output_np = np.array(output)
            output_np = output_np.reshape(1, 20, -1) # (1, 20, 540)
            data_ds  = np.concatenate((data_ds, output_np), axis=0) # data_ds.shape=(2503, 20, 540)
        
        np.save( dataset_path + 'processed_data_ds/' + 'sub{}_features'.format( usr_idx ), data_ds )
        np.save( dataset_path + 'processed_data_ds/' + 'sub{}_labels'.format( usr_idx ), label )
        print( "sub{} finished".format( usr_idx) )
        
        
        
def preprocess_pamap2( ):
    dataset_path    = ''
    channel_num     = 27
    
    if os.path.exists( dataset_path + 'processed_data/' ):
        shutil.rmtree( dataset_path + 'processed_data/' )
    os.mkdir( dataset_path + 'processed_data/' )

    lowerBound = np.array( [
        -18.1809,       -10.455566,     -7.7453649,         
        -3.9656347,     -2.54338,       -4.6695066,
        -40.503183,     -71.010566,     -68.132566,
        
        -4.0144814,     -2.169227,      -10.6296,       
        -1.0943914,     -1.7640771,     -0.85873557,    
        -35.379757,     -67.172728,     -42.7236,
        
        -1.540593,      -15.741095,     -12.220085,          
        -3.2952385,     -2.2376485,     -4.7753095,
        -92.0835,       -54.226,        -38.719725] )

    upperBound = np.array( [
        8.4245698,      18.942083,      10.753683,             
        4.0320432,      2.9766798,      4.4932249, 
        68.037749,      41.263183,      28.875164,
        
        4.8361714,      21.236957,      9.54976,                 
        1.13992,        1.7279,         0.84088028, 
        40.9678,        15.5543,        53.282157,
        
        26.630485,      24.1561,        6.9268395,              
        3.505307,       1.5628585,      6.4969625,
        5.87456,        44.9736,        61.16018] )

    file_list = [
        'subject101.dat', 'subject102.dat', 'subject103.dat', 'subject104.dat',
        'subject105.dat', 'subject106.dat', 'subject107.dat', 'subject108.dat'  ]

    for usr_idx in range(len(file_list)):

        file    = dataset_path + 'Protocol/' + file_list[usr_idx]
        data    = np.loadtxt( file ) # (376417, 54)

        label   = data[:,1].astype( int )
        label[label == 0]   = -1
        label[label == 1]   = 0         # lying
        label[label == 2]   = 1         # sitting
        label[label == 3]   = 2         # standing
        label[label == 4]   = 3         # walking
        label[label == 5]   = 4         # running
        label[label == 6]   = 5         # cycling
        label[label == 7]   = 6         # nordic walking
        label[label == 12]  = 7         # ascending stairs
        label[label == 13]  = 8         # descending stairs
        label[label == 16]  = 9         # vacuum cleaning
        label[label == 17]  = 10        # ironing
        label[label == 24]  = 11        # rope jumping

        # fill missing values
        #valid_idx   = np.concatenate( (np.arange(4, 16), np.arange(21, 33), np.arange(38, 50)), axis = 0 )
        valid_idx   = np.concatenate( (np.arange(4, 7), np.arange(10, 16), np.arange(21, 24),np.arange(27, 33), np.arange(38, 41), np.arange(44, 50)), axis = 0 )
        data        = data[ :, valid_idx ] # (376417, 27)
        data        = np.array( [Series(i).interpolate() for i in data.T] ).T

        # min-max normalization
        diff = upperBound - lowerBound
        data = 2 * (data - lowerBound) / diff - 1

        data[ data > 1 ]    = 1.0
        data[ data < -1 ]   = -1.0

        # sliding window
        data    = sliding_window( data, (FLAGS.length, channel_num), (FLAGS.overlap, 1) ) # (3763, 200, 27)
        label   = sliding_window( label, FLAGS.length, FLAGS.overlap )
        label   = stats.mode( label, axis=1 )[0][:,0]

        # remove non-interested time windows (label==-1)
        invalid_idx = np.nonzero( label < 0 )[0]
        data        = np.delete( data, invalid_idx, axis=0 )   # (2503, 200, 27)
        label       = np.delete( label, invalid_idx, axis=0 )  # (2503,)
    
        
        np.save( dataset_path + 'processed_data/' + 'sub{}_features'.format( usr_idx ), data )
        np.save( dataset_path + 'processed_data/' + 'sub{}_labels'.format( usr_idx ), label )
        print( "sub{} finished".format( usr_idx) )        
        

def main( _ ):
    #preprocess_opportunity()
    #preprocess_pamap2_ds()
    preprocess_pamap2()


if __name__ == '__main__':
    '''
    app.run(main)
    '''


    
    




