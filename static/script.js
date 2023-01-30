let fileToPredict = null;

$(document).ready(function () {
  var $ = jQuery;
  var domain = $('#hostUrl').text();

  $('#predict-button').click((e) => {
		e.preventDefault();
		e.stopPropagation();

    if (fileToPredict === null) {
      alert('Please upload or choose a fruit image!');
    }
    else {
      loadScreen(true);

      const url = domain + 'colorize';

      var formData = new FormData();
      const filename = fileToPredict.name;

      formData.append('grayscale_image', fileToPredict, filename);

      $.ajax(url, {
        type: 'POST',
        dataType: 'json',
        data: formData,
        cache: false,
        contentType: false,
        processData: false,
      })
      .done((response) => {
        if (response.status == 'ok') {
          if (response.data && response.data.colorized_image) {
            let imageSrc = 'data:image/png;base64,' + response.data.colorized_image;
            $('#colorized-image').attr('src', imageSrc);
          }

          if (response.data && response.data.grayscale_image) {
            let imageSrc = 'data:image/png;base64,' + response.data.grayscale_image;
            $('#grayscale-image').attr('src', imageSrc);
          }
        }
        loadScreen(false);
      })
      .fail((xhr, textStatus, errorThrown) => {
        loadScreen(false);

        let errMsg = 'Request failed with error: ' + xhr.responseText;
        console.log(errMsg);
        alert(errMsg);
      });
    }
  });


  	// File Upload Section
	$('#upload-form').click((e) => {
		e.preventDefault();
		e.stopPropagation();

		$('#file-input').trigger('click');
	})

	$('#file-input').change((e) => {
		e.preventDefault();
		e.stopPropagation();

		if (e.target.files && e.target.files.length > 0) {
			const files = e.target.files;
			storeFiles(files);
		}
	})

	 // preventing page from redirecting
	 $('#upload-form').on('dragover', e => {
		e.preventDefault();
		e.stopPropagation();
		
		$('#upload-form').attr('drop-active', 'True');
	 });

	$('#upload-form').on('dragleave', e => {
		e.preventDefault();
		e.stopPropagation();

		$('#upload-form').attr('drop-active', 'False');
	})
	
	 $('#upload-form').on('drop', e => {
		e.preventDefault();
		e.stopPropagation();

		$('#upload-form').attr('drop-active', 'False');

		const files = e.originalEvent.dataTransfer.files;
		storeFiles(files);
	});

  $('.select-image').each((i, obj) => {
    $(obj).click(async (e) => {
      let blob = await fetch(obj.src).then(r => r.blob());
      extension = obj.src.substring(obj.src.lastIndexOf('.'));
      blob.name = 'original_image' + extension;
      fileToPredict = blob;
      $('#selected-image').attr('src', obj.src);
    })
  })

	var storeFiles = (files) => {
		for (let i = 0; i < files.length; i++) {
			const file = files[i];
      fileToPredict = file;
			if (file.type !== 'image/jpeg' && file.type !== 'image/png') {
				
				let errMsg = "Please only upload JPEG or JPG or PNG file type!";
				alert(errMsg);
			}
			else {
        const image_url = (window.URL || window.webkitURL).createObjectURL(file);
        $('#selected-image').attr('src', image_url);
			}
		}
	}

  var loadScreen = (isLoading) => {
		if (isLoading) {
			$('#loading').attr('loading-active', 'True')
		}
		else {
			$('#loading').attr('loading-active', 'False')
		}
	}
});