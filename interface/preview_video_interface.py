import ipywidgets as wg
from IPython.display import display, HTML

videos_dir = "../Videos/"
data_dir = "../Data/"

def videoPreviewInterface(video_dropdown, json_dropdown, data_dropdown):

    def showVideo(video_dropdown, json_dropdown, data_dropdown):
        if(video_dropdown == "None"):
            print("Choose a video")
            return
        if(json_dropdown == "None" and data_dropdown == "None"):
            video_path = videos_dir + video_dropdown
        elif(data_dropdown == "None"):
            video_name = (video_dropdown).split(sep='.')[0]
            file_dir = data_dir + video_name + '/'
            file_name = (json_dropdown).split(sep='.')[0]
            video_path = file_dir + file_name + '.mp4'
        else:
            video_name = (video_dropdown).split(sep='.')[0]
            file_dir = data_dir + video_name + '/'
            file_name = (data_dropdown).split(sep='.')[0]
            video_path = file_dir + file_name + '.mp4'
        display(HTML("""<video width="640" height="360" controls="">
                    <source src="{0}"></video>""".format(video_path)))
    
    video_wg = wg.interactive_output(showVideo, {"video_dropdown":video_dropdown,
                                                "json_dropdown":json_dropdown,
                                                "data_dropdown":data_dropdown})
    hbox = wg.HBox([video_dropdown, json_dropdown, data_dropdown])
    video_menu = wg.VBox([hbox, video_wg])
    return video_menu