var app = new Vue({
	el: '#app',
	data: {
		query: '',
		state: null,
	},
	created: function() {
		setInterval(this.fetchState, 1000);
	},
	methods: {
		fetchState: function() {
			$.get('/state', (state) => {
				this.state = state;
			});
		},
		submitQuery: function() {
			this.state = null;
			$.post('/exec', {'query': this.query});
		},
	},
});
