#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    size = len(predictions)
    cleanSize = int(size * .1)
    bigError = [0] * cleanSize
    
    print "size: ", size
    print "cleanSize", cleanSize

    for x in range(0, size):

        errorVal = abs(predictions[x] - net_worths[x]) 
        
        # set value of cleaned data
        cleaned_data.append((ages[x], net_worths[x], errorVal))

        # keep track of the 10 largest error values
        if errorVal >= bigError[0]:
            bigError[0] = errorVal
            bigError = sorted(bigError)

    print bigError

    # remove biggest errors from list
    for y in range (0, cleanSize):
        for z in xrange (0, len(cleaned_data)):
            if cleaned_data[z][2] == bigError[y]:
                del cleaned_data[z]
                break
    
    return cleaned_data

