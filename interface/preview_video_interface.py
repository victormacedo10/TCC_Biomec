import ipywidgets as wg
from IPython.display import display, HTML

def videoPreviewInterface(video_dropdown):

    def showVideo(video_dropdown):
        if(video_dropdown == "None"):
            print("Choose a video")
            return
        video_path = videos_dir + video_dropdown
        display(HTML("""<video width="640" height="360" controls="">
                    <source src="{0}"></video>""".format(video_path)))
    
    video_wg = wg.interactive_output(showVideo, {"video_dropdown":video_dropdown})
    video_menu = wg.VBox([video_dropdown, video_wg])
    return video_menu