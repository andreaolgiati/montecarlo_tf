from montecarlo.rules.rule import Rule
from PIL import Image
import io
import numpy as np
import base64


class Luminance(Rule):
    def __init__(self, endpoint, min_threshold=0, max_threshold=1.0):
        super().__init__(endpoint, other_periods=None)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.logger.info("Luminance rule created.")

    @staticmethod
    def load_trace_data( trace ):
        inference_size = (224, 224) 
        raw_base64 = trace['capturedData']['0']['data']
        raw_decoded = base64.b64decode(raw_base64)
        raw_asbytes = io.BytesIO(raw_decoded)
        byteImg = Image.open(raw_asbytes)
        byteImg = Image.Image.resize(byteImg, size=inference_size)
        pix = np.array(byteImg)
        return (trace['eventId'], trace['inferenceTime'], pix, trace['capturedData']['2']['data'])


    def invoke_at_period(self, start_time, end_time, traces, storage_handler=None, **kwargs):
        luminances = []
        succeed, fail = 0, 1
        print( 'Trace')
        for trace in traces:
            try:
                event_id, event_time, pix, res = self.load_trace_data(trace)
                # luminance is the average pixel value, normalized to 0..1
                luminance = np.sum(pix)/np.product(pix.shape)/255    
                self.logger.info(f'ID={event_id}, TIME={event_time}, LUMINANCE={luminance}')
                res = self.check_luminance(luminance)
                if res:
                    succeed += 1
                else:
                    fail += 1
                luminances.append(luminance)
            except:
                raise
        self.logger.info(f'Period {start_time}-->{end_time} had {len(traces)} images, {succeed} had valid luminance and {fail} did not')
        return True if fail==0 else False

    # Return True if tensor contains all zeros (or above certain threshold)
    def check_luminance(self, lum):
        return (lum>self.min_threshold) and (lum<self.max_threshold)