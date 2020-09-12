import base64
import io


def get_base64(plt, tight=None):
    pic_IObytes = io.BytesIO()
    plt.savefig(pic_IObytes,  format='png', bbox_inches=tight)
    pic_IObytes.seek(0)
    return base64.b64encode(pic_IObytes.read())
