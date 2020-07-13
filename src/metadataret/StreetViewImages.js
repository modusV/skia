
function CarsHeadingCsv(options) {

    if (!options || !options.file) {
        throw new Error("Must provide file");
    }

    var sv_service = new google.maps.StreetViewService();
    var csv_data = [];
    var returned_calls = 0;
    var data_not_found = 0;


    wrapperFunc(options.file).then((result) => {
        
        var file_content = result;
        points = file_content.toString().split("\n");
        for(var p in points) {
            
            var lat_long = points[p].split(",");
            lat = lat_long[0];
            long = lat_long[1];
            var loc = new google.maps.LatLng(lat, long);
            sv_service.getPanorama({
                location: loc,
                radius: 50,
                source: google.maps.StreetViewSource.OUTDOOR,
                preference: google.maps.StreetViewPreference.BEST
            }, processSVData);
  
            function processSVData(data, status) {

                if (status === 'OK') {
                    
                    pano = data.location.pano;
                    date = data.time.filter((entry) => {
                            return entry.pano == pano    
                            }
                        );
                    
                    console.log(date);
                    if(date != undefined) {
                        date_string = date[0]['Cg'].mmddyyyy();
                        csv_data.push([date_string, pano, returned_calls, data.location.latLng.lat(), data.location.latLng.lng(), data.tiles.centerHeading]);
                    } else {
                        csv_data.push(['na', pano, returned_calls, data.location.latLng.lat(), data.location.latLng.lng(), data.tiles.centerHeading]);
                    }

                } else {
                    data_not_found += 1;
                    console.error('Street View data not found for this location.');
                }
                returned_calls += 1;
                if(returned_calls == points.length) {
                    download_csv(csv_data);
                }
            }
        }


    })


    function download_csv(data) {
        var csv = 'Date,PanoId,NumId,Lat,Long,Heading\n';
        data.forEach(function(row) {
                csv += row.join(',');
                csv += "\n";
        });
     
        var hiddenElement = document.createElement('a');
        hiddenElement.href = 'data:text/csv;charset=utf-8,' + encodeURI(csv);
        hiddenElement.target = '_blank';
        hiddenElement.download = 'cars_heading.csv';
        hiddenElement.click();
    }
    

    async function wrapperFunc(file) {
        try {
            let r1 = await promiseFile(file);
            return r1;     // this will be the resolved value of the returned promise
        } catch(e) {
            console.log(e);
            throw e;      // let caller know the promise was rejected with this reason
        }
    }
    
    async function promiseFile(file) {
        let a = new Promise((resolve, reject) => {
            var fr = new FileReader();  
            fr.onload = () => {
              resolve(fr.result)
            };
            fr.readAsText(file);
        });
        return a;
    }

    Date.prototype.mmddyyyy = function() {
        var mm = this.getMonth() + 1; // getMonth() is zero-based
        var dd = this.getDate();
      
        return [(mm>9 ? '' : '0') + mm,
                (dd>9 ? '' : '0') + dd,
                this.getFullYear()
               ].join('/');
      };
}

window.CarsHeadingCsv = CarsHeadingCsv;

