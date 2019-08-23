from montecarlo.rules.rule import Rule
#from montecarlo.analysis.utils import parse_list_from_str
import numpy as np
import base64


class Luminance(Rule):
    def __init__(self, base_period, min_threshold=0, max_threshold=1.0):
        super().__init__(base_period, other_period=None)
        self.min_threshold = float(min_threshold)
        self.max_threshold = float(max_threshold)
        self.logger.info("Luminance rule created.")

    def invoke_at_period(self, period, traces, storage_handler=None, **kwargs):
        inference_size = (224, 224) 
        luminances = []
        succeed, fail = 0, 0
        for trace in traces:
            # The raw payload, expressed as base64
            raw_base64 = trace['capturedData']['0']['data']
            # These bytes describe a jpeg image
            raw_decoded = base64.b64decode(raw_base64)
            try:
                raw_asbytes = io.BytesIO(raw_decoded)
                # Decoded image
                byteImg = Image.open(raw_asbytes)
                pix = np.array(byteImg)
                # now we resize to 224x224
                byteImg = Image.Image.resize(byteImg, size=inference_size)
                pix = np.array(byteImg)
                # luminance is the average pixel value, normalized to 0..1
                luminance = np.sum(pix)/np.product(pix.shape)/255    
                res = check_luminance(luminance)
                if res:
                    succeed += 1
                else:
                    fail += 1
                luminances.append(luminance)
            except:
                raise
        self.logger.info(f'Period {period} had {len(traces)} images, {succeed} had valid luminance and {fail} did not')
        return True if fail==0 else False

    # Return True if tensor contains all zeros (or above certain threshold)
    def check_luminance(self, lum):
        return (lum>self.min_threshold) and (lum<self.max_threshold)