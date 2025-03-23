import React, { useState } from "react";

function AutismScreening() {
    const [formData, setFormData] = useState({
        A1_score: 0,
        A2_score: 0,
        age: 25,
        gender: "male"
    });
    
    const [result, setResult] = useState("");

    const handleChange = (e) => {
        setFormData({
            ...formData,
            [e.target.name]: e.target.value
        });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();

        try {
            const response = await fetch("http://127.0.0.1:5000/predict_model", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();
            setResult(data.prediction);
        } catch (error) {
            console.error("Error:", error);
        }
    };

    return (
        <div>
            <h2>Autism Screening Form</h2>
            <form onSubmit={handleSubmit}>
                <label>A1 Score:</label>
                <input type="number" name="A1_score" value={formData.A1_score} onChange={handleChange} />

                <label>A2 Score:</label>
                <input type="number" name="A2_score" value={formData.A2_score} onChange={handleChange} />

                <label>Age:</label>
                <input type="number" name="age" value={formData.age} onChange={handleChange} />

                <label>Gender:</label>
                <select name="gender" value={formData.gender} onChange={handleChange}>
                    <option value="male">Male</option>
                    <option value="female">Female</option>
                </select>

                <button type="submit">Predict</button>
            </form>

            {result && <h3>Prediction: {result}</h3>}
        </div>
    );
}

export default AutismScreening;
