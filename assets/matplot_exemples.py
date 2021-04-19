'''
    # display the median of frames
    fig, axs = plt.subplots(2,2)
    
    axs[0, 0].imshow(f_median[0], cmap=plt.cm.seismic, origin='lower')
    axs[0, 0].set_title("median wave length 1")
    im2 = axs[0, 1].imshow(f_median[1], cmap=plt.cm.seismic, origin='lower')
    axs[0, 1].set_title("median wave length 2")
    axs[1, 0].imshow(res[0], cmap=plt.cm.seismic, origin='lower')
    axs[1, 0].set_title("res wave length 1")
    im4 = axs[1, 1].imshow(res[1], cmap=plt.cm.seismic, origin='lower')
    axs[1, 1].set_title("res wave length 2")
    
    #fig.subplots_adjust(right=1.9)
    #pos1 = fig.add_axes([0.92, 0.41, 0.015, 0.77])#位置[左,下,右,上]
    cb = fig.colorbar(im2, ax=axs[0,1])
    
    #pos2 = fig.add_axes([0.92, 0.11, 0.015, 0.30])#位置[左,下,右,上]
    cb = fig.colorbar(im4, ax=axs[1,1])

    plt.show()
    '''
