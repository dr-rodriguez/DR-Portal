from flask import Flask, render_template, request, redirect, make_response
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool, OpenURL, TapTool
import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.wcs import WCS
from bokeh.palettes import Spectral4
from astroquery.mast import Observations
from bokeh.models import DataTable, TableColumn
from bokeh.layouts import column, widgetbox

app_portal = Flask(__name__)

blc, trc, im = None, None, None

# Redirect to the main page
@app_portal.route('/')
@app_portal.route('/index')
@app_portal.route('/index.html')
def app_home():
    load_image()

    return render_template('index.html')


def load_image():
    global blc
    global trc
    global im
    image_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits', cache=True)
    hdu_list = fits.open(image_file)
    im = hdu_list[0].data
    ny, nx = im.shape
    w = WCS(hdu_list[0].header)

    pixcrd = np.array([[0, 0], [nx, ny]], np.float_)
    world = w.wcs_pix2world(pixcrd, 0)
    blc, trc = world  # bottom left corner, top right corner


def make_base_bokeh():
    global blc
    global trc
    global im

    if blc is None or trc is None or im is None: load_image()

    tools = 'pan, box_zoom, wheel_zoom, save, reset, resize'
    p = figure(plot_width=700, x_range=(blc[0], trc[0]), y_range=(blc[1], trc[1]))

    # Get the physical ranges
    nx_wcs = abs(blc[0] - trc[0])
    ny_wcs = abs(blc[1] - trc[1])
    p.plot_height = int(p.plot_width * ny_wcs / nx_wcs)

    # must give a vector of image data for image parameter
    # Documentation for p.image https://bokeh.pydata.org/en/latest/docs/reference/models/glyphs/image.html
    p.image(image=[im], x=blc[0], y=blc[1], dw=nx_wcs, dh=ny_wcs, palette='Greys256')

    # Axis labels
    p.xaxis.axis_label = 'RA (deg)'
    p.yaxis.axis_label = 'Dec (deg)'

    return p


# Page with a text box to take the SQL query
@app_portal.route('/catalogs', methods=['GET', 'POST'])
def app_catalogs():
    global blc
    global trc
    global im
    if blc is None or trc is None or im is None: load_image()

    from astroquery.mast import Catalogs

    searchString = '{} {}'.format(np.mean([blc[0], trc[0]]), np.mean([blc[1], trc[1]]))
    catalogData = Catalogs.query_object(searchString, radius=0.2, catalog="GAIAdr2")

    # get plot
    p = make_base_bokeh()

    source = ColumnDataSource(catalogData.to_pandas())
    p.scatter('ra', 'dec', source=source, legend="GAIA DR2", alpha=0.7, size=10)

    # Add hover tooltip for GAIA data
    tooltip = [("RA", "@ra"), ("Dec", "@dec"), ("Desig.", "@designation"), ("parallax", "@parallax"),
               ("phot_g_mean_mag", "@phot_g_mean_mag")]
    p.add_tools(HoverTool(tooltips=tooltip))

    p.legend.click_policy = "hide"

    # Table data
    columns = []
    for col in catalogData.to_pandas().columns:
        if col not in ('ra', 'dec', 'designation', 'parallax'):
            continue
        columns.append(TableColumn(field=col, title=col))
    data_table = DataTable(source=source, columns=columns, width=1200, height=280)

    # Fails to load anything
    # script, div_dict = components({'plot': p, 'table': widgetbox(data_table)})
    # return render_template('catalogs.html', script=script, div=div_dict)

    # Fails to load table
    # script1, div1 = components(p)
    # script2, div2 = components(widgetbox(data_table))
    # return render_template('catalogs.html', script1=script1, div1=div1, script2=script2, div2=div2)

    # No table
    script, div = components(p)
    return render_template('base.html', script=script, plot=div)


@app_portal.route('/caom', methods=['GET', 'POST'])
def app_caom():
    global blc
    global trc
    global im
    if blc is None or trc is None or im is None: load_image()

    searchString = '{} {}'.format(np.mean([blc[0], trc[0]]), np.mean([blc[1], trc[1]]))
    obsTable = Observations.query_region(searchString, radius=0.2)

    obsDF = obsTable.to_pandas()
    obsDF = obsDF[(obsDF['obs_collection'].isin(['HST', 'SWIFT'])) &
                  (obsDF['instrument_name'].isin(['WFC3/IR', 'ACS/WFC', 'STIS/CCD']))]

    obsDF['coords'] = obsDF.apply(lambda x: parse_s_region(x['s_region']), axis=1)
    obsDF['coords'].head()

    # get plot
    p = make_base_bokeh()

    # Loop over instruments, coloring each separately
    for ins, color in zip(obsDF['instrument_name'].unique(), Spectral4):
        ind = obsDF['instrument_name'] == ins
        # Add patches with the observation footprings
        patch_xs = [c['ra'] for c in obsDF['coords'][ind]]
        patch_ys = [c['dec'] for c in obsDF['coords'][ind]]

        data = {'x': patch_xs, 'y': patch_ys, 'obs_collection': obsDF['obs_collection'][ind],
                'instrument_name': obsDF['instrument_name'][ind], 'obs_id': obsDF['obs_id'][ind],
                'target_name': obsDF['target_name'][ind], 'proposal_pi': obsDF['proposal_pi'][ind]}
        p.patches('x', 'y', source=data, legend=ins,
                  fill_color=color, fill_alpha=0.1, line_color="white", line_width=0.5)

    # Add hover tooltip for MAST observations
    tooltip = [("obs_collection", "@obs_collection"),
               ("instrument_name", "@instrument_name"),
               ("obs_id", "@obs_id"),
               ("target_name", "@target_name"),
               ('proposal_pi', '@proposal_pi')]
    p.add_tools(HoverTool(tooltips=tooltip))

    p.legend.click_policy = "hide"

    script, div = components(p)

    return render_template('base.html', script=script, plot=div)


def parse_s_region(s_region):
    ra = []
    dec = []
    counter = 0
    for elem in s_region.strip().split():
        try:
            value = float(elem)
        except ValueError:
            continue
        if counter % 2 == 0:
            ra.append(value)
        else:
            dec.append(value)
        counter += 1

    return {'ra': ra, 'dec': dec}

